module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @empty_cp_group() {
    %group = ttg.async_commit_group
    %wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
}
