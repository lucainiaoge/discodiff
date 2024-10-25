from config.base.attrdict import AttrDict

config_training_default = AttrDict(
    train_text_clap_load_prob=0.5,
    train_dataset_size=51200,  # dummy dataset size
    train_batch_size=64,

    val_dataset_size=1,  # dummy dataset size
    val_batch_size=1,

    ema_decay=0.995,
    learning_rate=5e-5,
    lr_warmup_steps=500,

    checkpoint_every=16000,
    demo_every=4000,
    max_epochs=1000,
    save_top_k=11,

    train_primary_prob=0.5,
    load_audio_clap_prob=0.5,
    clap_cond_drop_prob=0.5,
    t5_cond_drop_prob=0.1,
    key_cond_drop_prob=0.4,
)

config_training_debug = AttrDict(
    train_text_clap_load_prob=0.5,
    train_dataset_size=12800,  # dummy dataset size
    train_batch_size=128,

    val_dataset_size=4,  # dummy dataset size
    val_batch_size=4,

    ema_decay=0.995,
    learning_rate=5e-5,
    lr_warmup_steps=100,

    checkpoint_every=4000,
    demo_every=800,
    max_epochs=5000,
    save_top_k=1,

    train_primary_prob=0.5,
    load_audio_clap_prob=0.5,
    clap_cond_drop_prob=0.5,
    t5_cond_drop_prob=0.1,
    key_cond_drop_prob=0.4,
)