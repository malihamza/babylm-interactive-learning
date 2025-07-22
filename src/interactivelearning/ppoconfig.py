from trl import PPOConfig

class CustomPPOConfig(PPOConfig):
    def __init__(
        self,
        *args,
        token_limit=None,
        checkpoint_interval=10000,
        output_min_length=4,
        output_max_length=16,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.token_limit = token_limit
        self.checkpoint_interval = checkpoint_interval
        self.output_min_length = output_min_length
        self.output_max_length = output_max_length
