from methods.inflora import InfLoRA
from methods.kernel_head import KernelHeadMixin


class InfLoRA_KH(KernelHeadMixin, InfLoRA):
    """InfLoRA with an additional analytic kernel classifier head.

    Training is identical to InfLoRA (the per-task linear heads are still used to
    train the adapters). After each task the kernel head absorbs that task's
    training features; eval_task reports both the original concatenated-head
    accuracy (CNN ...) and the kernel-head accuracy (KH ...) in the log.
    """

    def __init__(self, args):
        super().__init__(args)
        self._init_kernel_head(args)

    def incremental_train(self, data_manager):
        super().incremental_train(data_manager)
        self._kh_after_task(data_manager)

    def eval_task(self):
        results = super().eval_task()
        self._kh_eval(self.test_loader)
        return results
