import orbax.checkpoint as ocp
from etils import epath


def write_model(params, step, dir, init: bool = False):
    if init:
        dir = ocp.test_utils.erase_and_create_empty(dir)
        return
    else:
        dir = epath.Path(dir)
    
    options = ocp.CheckpointManagerOptions(max_to_keep=1,
                                            create=True)
    mngr = ocp.CheckpointManager(directory=dir,
                                    options=options)
    saved = mngr.save(step=step,
                        args=ocp.args.StandardSave(params))
    mngr.wait_until_finished()
    return
    

def load_model(step, dir):
    options = ocp.CheckpointManagerOptions(max_to_keep=1,
                                            create=True)
    mngr = ocp.CheckpointManager(directory=dir,
                                    options=options,
                                    item_handlers=ocp.StandardCheckpointHandler())
    return mngr.restore(step=step)