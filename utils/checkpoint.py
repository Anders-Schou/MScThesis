import jax
import jax.numpy as jnp
import flax.training.train_state
from flax.training import checkpoints
import orbax.checkpoint as ocp
import optax

# optax.chain
# optax.clip_by_global_norm
# optax.multi_transform
# optax.adam

# self.params = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None)
# orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
# checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
#                             target=self.params,
#                             step=iter,
#                             overwrite=False,
#                             orbax_checkpointer=orbax_checkpointer)
# checkpoints.save_checkpoint(ckpt_dir="/zhome/e8/9/147091/MSc/results/models",
#                                 target=params,
#                                 step=step,
#                                 overwrite=False,
#                                 orbax_checkpointer=orbax_checkpointer)

# def write_model(params, step, settings=None):
#     dir = ocp.test_utils.erase_and_create_empty("/zhome/e8/9/147091/MSc/tests/dir_test")
#     options = ocp.CheckpointManagerOptions(max_to_keep=2,
#                                            create=True)
#     mngr = ocp.CheckpointManager(directory="/zhome/e8/9/147091/MSc/tests/dir_test",
#                                  options=options)
#     mngr.save(step=step,
#               args=ocp.args.StandardSave(params))
#     # mngr.wait_until_finished()
    

# def load_model(step):
#     options = ocp.CheckpointManagerOptions(max_to_keep=2,
#                                            create=True)
#     mngr = ocp.CheckpointManager(directory="/zhome/e8/9/147091/MSc/tests/dir_test",
#                                  options=options,
#                                  item_handlers=ocp.StandardCheckpointHandler())
#     return mngr.restore(step=step)

# ocp.args.StandardRestore
# ocp.args.Composite
# ocp.args.JsonSave
# checkpoints.restore_checkpoint
# ocp.utils.checkpoint_steps



TEST_DIR = "/zhome/e8/9/147091/MSc/tests/dir_test"

def write_model(params, step, settings=None):
    dir = ocp.test_utils.erase_and_create_empty(TEST_DIR)
    options = ocp.CheckpointManagerOptions(max_to_keep=2,
                                            create=True)
    mngr = ocp.CheckpointManager(directory=dir,
                                    options=options)
    saved = mngr.save(step=step,
                        args=ocp.args.StandardSave(params))
    mngr.wait_until_finished()
    return
    

def load_model(step):
    options = ocp.CheckpointManagerOptions(max_to_keep=2,
                                            create=True)
    mngr = ocp.CheckpointManager(directory=TEST_DIR,
                                    options=options,
                                    item_handlers=ocp.StandardCheckpointHandler())
    return mngr.restore(step=step)