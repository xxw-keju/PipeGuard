    @T.prim_func
    def main():
        with T.Kernel(1, threads=32):
            T.tvm_storage_sync("shared.dyn")
