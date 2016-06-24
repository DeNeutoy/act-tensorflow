import time
import numpy as np
import reader


def run_epoch(session, m, data, eval_op, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0

    for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size, m.num_steps)):
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y})
        

        # if verbose and step % 100 == 0: print('you successfully completed one entire batch -- cost', cost)
        if verbose: print('you successfully completed one entire batch -- cost', cost)
        costs += cost
        iters += m.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * m.batch_size / (time.time() - start_time)))


    return (costs / iters)



