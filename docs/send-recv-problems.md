  - dist.batch_isend_irecv 会绕过当前 Python wrapper，先直接 group._get_backend(mcpu)，所以首次 mcpu batch P2P 失败。更重要的是，我临时尝试只补 lazy 注册后发现会引出更深的问题：C++ P2P 当前
    把 blocking Gloo send/recv 串到同一个 mcpu stream worker，batch 中双方都先 send 再 recv 时可能互等死锁。

  我刚实测了当前安装版本：

  - CPU/Gloo：
      - 两个 rank 都 isend(); irecv(); wait()：正常完成。
      - 两个 rank 都 batch_isend_irecv([isend, irecv])：正常完成。

  - MCPU 当前实现：
      - 两个 rank 都连续 dist.isend(...); dist.irecv(...)：30s 超时。
      - dist.batch_isend_irecv([isend, irecv])：还没进到死锁，先报 No backend type associated with device type mcpu。

  所以不是“两个通信可能不在一个 stream 上所以没事”。当前普通连续 P2P 已经会卡。根因仍是实现把 blocking Gloo send/recv 放进 mcpu stream worker，双方都先 send 时 worker 被 send 占住，recv 没
  机会提交。

  可行修法不是换 stream，而是改成 host progress：

  - send/recv/recvAnysource 不进入 mcpu stream worker 执行 blocking Gloo work。
  - 提交时记录当前 mcpu stream 的 ready event。
  - 后台 host 线程等待这个 event 完成，再进入 KernelTaskScope/解除 memory protection，创建 CPU view，调用底层 Gloo P2P work 并 wait。
  - 完成后标记自定义 Work 完成；Work.wait() 对 P2P 应该等待 host 线程完成，而不是只插 stream event。
