import os
import glob
import shutil
import numpy as np
import multiprocessing
import gc
import random
import heapq
import traceback
import json
from tqdm import tqdm

# 必须使用完整修复版 indexed_dataset.py
from indexed_dataset import IndexedDataset, _IndexWriter

# ================= 配置区域 =================
TARGET_SPLITS = 2 # 目标分片数
MAX_CONCURRENCY = 32  # 并发数
CHUNK_TARGET_SIZE = 100 * 1024 ** 2  # 块
ALLOCATION_FILE = '/mnt/data/export/allocation.json'  # 持久化分配文件名称


# ===========================================

def get_output_path_for_pid(pid, output_dirs):
    """根据 partition_id 轮询分配输出路径"""
    return output_dirs[pid % len(output_dirs)]


def get_file_chunks(file_path):
    """扫描源文件并切成大块 (保持不变)"""
    chunks = []
    prefix = os.path.splitext(file_path)[0]
    bin_path = prefix + ".bin"
    if not os.path.exists(bin_path): return []

    try:
        ds = IndexedDataset(prefix, mmap=True)
        sizes = ds.sequence_lengths
        bin_size = os.path.getsize(bin_path)
        total_tokens = np.sum(sizes, dtype=np.int64)
        elem_size = 2 if total_tokens == 0 else int(bin_size // total_tokens)

        pointers = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64))) * elem_size
        total_docs = len(ds)
        dtype_code = ds.index.dtype.__name__
        is_multi = getattr(ds.index, 'sequence_modes', None) is not None

        cur = 0
        while cur < total_docs:
            start_byte = pointers[cur]
            target_byte = start_byte + CHUNK_TARGET_SIZE
            nxt = np.searchsorted(pointers, target_byte)
            if nxt >= total_docs: nxt = total_docs
            if nxt <= cur: nxt = cur + 1

            end_byte = pointers[nxt]
            chunk_len = nxt - cur

            chunks.append({
                'id': f"{prefix}_{cur}",
                'src_path': prefix,
                'start_doc': int(cur),
                'end_doc': int(nxt),
                'start_byte': int(start_byte),
                'end_byte': int(end_byte),
                'byte_size': int(end_byte - start_byte),
                'num_docs': int(nxt - cur),
                'dtype': dtype_code,
                'multi': is_multi,
                'first_len': int(sizes[cur]) if chunk_len > 0 else 0,
                'last_len': int(sizes[nxt - 1]) if chunk_len > 0 else 0
            })
            cur = nxt

        if hasattr(ds.index, 'dealloc'): ds.index.dealloc()
        del ds
        return chunks
    except Exception as e:
        print(f"[Skip] {file_path}: {e}")
        return []


def load_chunk_sizes(task):
    """(保持不变)"""
    try:
        ds = IndexedDataset(task['src_path'], mmap=True)
        sizes = ds.sequence_lengths
        result = np.copy(sizes[task['start_doc']: task['end_doc']])
        if hasattr(ds.index, 'dealloc'): ds.index.dealloc()
        del ds
        return result
    except:
        return None


def clean_bad_outputs(output_dirs):
    """[修改] 扫描所有输出路径，自动清理历史坏文件"""
    print("\n正在自动清理历史损坏的分区...")
    # 遍历所有配置的输出路径
    for out_dir in output_dirs:
        if not os.path.exists(out_dir): continue
        pattern = os.path.join(out_dir, "data_*_final")
        for folder in glob.glob(pattern):
            prefix = os.path.join(folder, os.path.basename(folder))
            idx_file = prefix + ".idx"
            bin_file = prefix + ".bin"
            # 检查文件完整性
            if not (os.path.exists(idx_file) and os.path.exists(bin_file)):
                continue
            try:
                test_ds = IndexedDataset(prefix, mmap=True)
                del test_ds
            except Exception as e:
                print(f"检测到损坏分区，自动删除 → {folder}")
                shutil.rmtree(folder, ignore_errors=True)


def identify_used_chunks(all_chunks, output_dirs):
    """[修改] 在所有输出路径中进行指纹比对"""
    print("\n开始指纹比对，识别已完成块...")

    chunk_lookup = {}
    for c in all_chunks:
        key = c['first_len']
        chunk_lookup.setdefault(key, []).append(c)

    used_ids = set()

    # [修改] 扫描所有 output_dirs
    existing_idx = []
    for out_dir in output_dirs:
        if os.path.exists(out_dir):
            existing_idx.extend(glob.glob(os.path.join(out_dir, "*", "*_final.idx")))

    existing_idx = sorted(existing_idx)

    if not existing_idx:
        print("未发现有效输出文件，将全量生成")
        return used_ids

    print(f"发现 {len(existing_idx)} 个输出文件 (跨越多个路径)，开始指纹验证...")
    for idx_file in existing_idx:
        prefix = os.path.splitext(idx_file)[0]
        try:
            ds = IndexedDataset(prefix, mmap=True)
            big_sizes = ds.sequence_lengths
            if len(big_sizes) == 0:
                raise ValueError("空文件")
        except Exception as e:
            print(f"无效文件（已跳过）: {os.path.basename(idx_file)} → {e}")
            continue

        cursor = 0
        total = len(big_sizes)
        # 获取文件名用于显示日志
        fname = os.path.basename(idx_file)

        pbar = tqdm(total=total, desc=f"验证 {fname}", leave=False)
        while cursor < total:
            start_val = big_sizes[cursor]
            possible_chunks = chunk_lookup.get(start_val, [])
            matched = False
            for cand in possible_chunks:
                if cand['id'] in used_ids:
                    continue
                if cursor + cand['num_docs'] > total:
                    continue
                if big_sizes[cursor + cand['num_docs'] - 1] != cand['last_len']:
                    continue
                source_sizes = load_chunk_sizes(cand)
                if source_sizes is None:
                    continue
                if np.array_equal(big_sizes[cursor: cursor + cand['num_docs']], source_sizes):
                    used_ids.add(cand['id'])
                    cursor += cand['num_docs']
                    pbar.update(cand['num_docs'])
                    matched = True
                    break

            if not matched:
                print(f"\n警告：{fname} 在位置 {cursor} 处无法匹配，停止验证此文件")
                break
        pbar.close()
        if cursor == total:
            print(f"成功验证完整文件: {fname}")
        else:
            print(f"部分验证: {fname} (匹配 {cursor}/{total})")
        del ds

    print(f"指纹比对完成，共识别 {len(used_ids)} 个已完成块")
    return used_ids


def worker_write(partition_id, tasks, output_root_path):
    """
    [深度优化版]
    1. 缓存源文件句柄，避免重复 open/close
    2. 使用 posix_fadvise 进行 IO 预取建议
    3. 增加周期性 fsync 防止内存脏页堆积导致的气泡
    """
    if not tasks:
        return f"P{partition_id} 无任务"

    # 关键：确保按源文件排序，最大化利用文件句柄缓存
    tasks.sort(key=lambda x: (x['src_path'], x['start_byte']))

    folder = f"data_{partition_id:02d}_final"
    path = os.path.join(output_root_path, folder)
    prefix = os.path.join(path, folder)
    out_bin = prefix + ".bin"
    out_idx = prefix + ".idx"

    if os.path.exists(out_bin) and os.path.exists(out_idx):
        return f"P{partition_id} 已存在，跳过"

    os.makedirs(path, exist_ok=True)

    # 配置：每写入多少字节强制刷盘一次 (建议 256MB - 512MB)
    SYNC_THRESHOLD = 256 * 1024 * 1024
    bytes_since_sync = 0

    try:
        # 使用较大的 buffer (1MB) 优化 USB 写入
        with open(out_bin, 'wb', buffering=1024 * 1024) as fout:
            total_bytes = sum(t['byte_size'] for t in tasks)
            pbar = tqdm(total=total_bytes, desc=f"P{partition_id}->{os.path.basename(output_root_path)}",
                        position=partition_id, leave=False, unit='B', unit_scale=True)

            all_sizes = []
            if not tasks: return "Empty Tasks"

            dtype_code = tasks[0]['dtype']
            is_multi = tasks[0]['multi']

            # === 优化核心：源文件句柄缓存变量 ===
            current_src_path = None
            src_f = None
            src_ds = None

            try:
                for task in tasks:
                    # 1. 检测是否需要切换源文件
                    if task['src_path'] != current_src_path:
                        # 关闭上一个文件
                        if src_f: src_f.close()
                        if src_ds and hasattr(src_ds.index, 'dealloc'): src_ds.index.dealloc()

                        # 打开新文件
                        current_src_path = task['src_path']
                        src_bin_path = current_src_path + ".bin"

                        # 仅加载 IndexedDataset 一次
                        src_ds = IndexedDataset(current_src_path, mmap=True)
                        # 保持 bin 文件开启
                        src_f = open(src_bin_path, 'rb')

                    # 2. 从缓存的 DS 获取 sizes (内存操作，极快)
                    # 注意：IndexedDataset 的 sequence_lengths 可能很大，这里只切片
                    # 为了安全，这里重新从 mmap 获取，开销很小
                    sizes = src_ds.sequence_lengths
                    chunk_sizes = np.copy(sizes[task['start_doc']: task['end_doc']])
                    all_sizes.append(chunk_sizes)

                    # 3. IO 预取建议 (针对机械硬盘的神器)
                    if hasattr(os, 'posix_fadvise'):
                        try:
                            # 告诉内核：我要顺序读这段数据了，请预读
                            os.posix_fadvise(src_f.fileno(), task['start_byte'], task['byte_size'],
                                             os.POSIX_FADV_SEQUENTIAL)
                            # 告诉内核：我也许很快会读它（Will Need）
                            os.posix_fadvise(src_f.fileno(), task['start_byte'], task['byte_size'],
                                             os.POSIX_FADV_WILLNEED)
                        except Exception:
                            pass

                    # 4. 执行写入 (sendfile 或 copyfileobj)
                    if hasattr(os, 'sendfile'):
                        offset = task['start_byte']
                        remain = task['byte_size']
                        while remain > 0:
                            # sendfile 是零拷贝，效率最高
                            sent = os.sendfile(fout.fileno(), src_f.fileno(), offset, remain)
                            if sent == 0: break
                            offset += sent
                            remain -= sent
                            pbar.update(sent)
                            bytes_since_sync += sent
                    else:
                        src_f.seek(task['start_byte'])
                        shutil.copyfileobj(src_f, fout, length=task['byte_size'])
                        pbar.update(task['byte_size'])
                        bytes_since_sync += task['byte_size']

                    # 5. 主动平滑刷盘 (消除气泡)
                    if bytes_since_sync >= SYNC_THRESHOLD:
                        fout.flush()
                        os.fsync(fout.fileno())
                        bytes_since_sync = 0
                        # 可选：告诉内核源文件的这段缓存可以丢弃了，释放内存
                        if hasattr(os, 'posix_fadvise'):
                            try:
                                os.posix_fadvise(src_f.fileno(), task['start_byte'], task['byte_size'],
                                                 os.POSIX_FADV_DONTNEED)
                            except:
                                pass

            finally:
                # 循环结束，清理最后一个打开的文件
                if src_f: src_f.close()
                if src_ds and hasattr(src_ds.index, 'dealloc'): src_ds.index.dealloc()

            pbar.close()

        # 生成 idx
        final_sizes = np.concatenate(all_sizes).astype(np.int32)
        del all_sizes
        gc.collect()

        dtype_np = getattr(np, dtype_code)
        doc_indices = list(range(len(final_sizes) + 1))

        with _IndexWriter(out_idx, dtype_np) as writer:
            writer.write(
                sequence_lengths=final_sizes.tolist(),
                sequence_modes=[0] * len(final_sizes) if is_multi else None,
                document_indices=doc_indices
            )
        return f"P{partition_id} Done (in {os.path.basename(output_root_path)})"

    except Exception as e:
        print(f"P{partition_id} 写入失败: {e}")
        traceback.print_exc()
        if os.path.exists(out_bin): os.remove(out_bin)
        if os.path.exists(out_idx): os.remove(out_idx)
        return f"P{partition_id} FAILED"


def main_resume(source_dirs, output_dirs_list):
    """[修改] 接收路径列表"""
    # 确保所有输出目录存在
    for d in output_dirs_list:
        os.makedirs(d, exist_ok=True)

    # 元数据默认存在第一个目录中，防止混乱
    meta_dir = output_dirs_list[0]
    allocation_path = os.path.join(meta_dir, ALLOCATION_FILE)

    print("=== 多路径并行 Shuffle 系统 ===")
    print(f"目标分片: {TARGET_SPLITS}")
    print(f"输出路径 ({len(output_dirs_list)}个):")
    for i, d in enumerate(output_dirs_list):
        print(f"  [{i}] {d}")
    print(f"元数据存储: {allocation_path}")

    # 1. 自动清理历史坏文件 (扫描所有路径)
    clean_bad_outputs(output_dirs_list)

    # 2. 扫描源文件
    all_idx_files = []
    for d in source_dirs:
        all_idx_files.extend(glob.glob(os.path.join(d, "*.idx")))

    print(f"\nStep 1: 扫描源文件 ({len(all_idx_files)} 个)...")
    all_chunks = []
    with multiprocessing.Pool(min(16, len(all_idx_files) or 1)) as pool:
        for chunks in tqdm(pool.imap_unordered(get_file_chunks, all_idx_files), total=len(all_idx_files)):
            all_chunks.extend(chunks)
    print(f"总块数: {len(all_chunks)}")

    # 3. 指纹去重 (扫描所有路径)
    used_ids = identify_used_chunks(all_chunks, output_dirs_list)
    remaining = [c for c in all_chunks if c['id'] not in used_ids]
    print(f"已完成块: {len(used_ids)}，剩余块: {len(remaining)}")

    if not remaining:
        print("所有数据已完成！")
        if os.path.exists(allocation_path):
            os.remove(allocation_path)
        return

    # 4. 分配任务
    if os.path.exists(allocation_path):
        print(f"加载历史分配文件: {allocation_path}")
        with open(allocation_path, 'r') as f:
            buckets = json.load(f)
        if len(buckets) != TARGET_SPLITS:
            print("历史分配不匹配，重新分配...")
            os.remove(allocation_path)
            buckets = None
    else:
        buckets = None

    if buckets is None:
        print("重新分配剩余块（负载均衡）...")
        random.shuffle(remaining)
        remaining.sort(key=lambda x: x['byte_size'], reverse=True)
        buckets = [[] for _ in range(TARGET_SPLITS)]
        heap = [(0, i) for i in range(TARGET_SPLITS)]
        heapq.heapify(heap)
        for chunk in remaining:
            load, pid = heapq.heappop(heap)
            buckets[pid].append(chunk)
            heapq.heappush(heap, (load + chunk['byte_size'], pid))

        with open(allocation_path, 'w') as f:
            json.dump(buckets, f)
        print(f"分配已保存到: {allocation_path}")

    # 5. 确定未完成的 partitions (检查其应该所在的具体路径)
    target_tasks = []  # (pid, target_path)

    for pid in range(TARGET_SPLITS):
        # 计算该 pid 应该分配到哪个路径
        target_path = get_output_path_for_pid(pid, output_dirs_list)
        folder = f"data_{pid:02d}_final"
        idx_path = os.path.join(target_path, folder, f"{folder}.idx")
        bin_path = os.path.join(target_path, folder, f"{folder}.bin")

        # 只要文件不完整，就加入任务队列
        if not (os.path.exists(idx_path) and os.path.exists(bin_path)):
            target_tasks.append((pid, target_path))
        else:
            # 双重检查：如果用户手动移动了文件位置导致 pid 和 path 不匹配，这里可能需要额外逻辑
            # 但通常假设用户不动生成好的文件
            pass

    if not target_tasks:
        print("所有分区已完成！")
        os.remove(allocation_path)
        return

    print(f"未完成分区数: {len(target_tasks)}")
    for pid, path in target_tasks:
        print(f"  -> P{pid} 将写入: {path}")

    # 6. 执行剩余
    print("\nStep 3: 开始多路径并发写入...")
    pool = multiprocessing.Pool(MAX_CONCURRENCY)
    results = []

    for pid, target_path in target_tasks:
        # 传递具体的 target_path
        results.append(pool.apply_async(worker_write, (pid, buckets[pid], target_path)))

    pool.close()
    pool.join()

    for res in results:
        print(res.get())

    print("\n任务完成！")


# ================== 配置你的路径 ==================
if __name__ == "__main__":
    SOURCE_DIRS = ["/share/nfs/data_02_final"]

    # [修改] 这里配置你的多个输出磁盘路径
    # 脚本会自动按顺序将 P0->Disk1, P1->Disk2, P2->Disk3, P3->Disk1 ... 循环分配
    OUTPUT_DIRS_LIST = [
        "/share/nfs/data_02_final/output_split"
    ]

    main_resume(SOURCE_DIRS, OUTPUT_DIRS_LIST)