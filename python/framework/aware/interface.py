import copy
import traceback
from framework.aware.simulator import AwareSimulator
from framework.aware.progressive_placer import ProgressivePlacer

__doc__ = "Interface from aware c++ to aware python"


class PyAwareInterface:
    """
    Interface from aware c++ to aware python
    """

    def __init__(self, config_params=None, simulator_params=None, networkx_graph=None):
        self.config_params = config_params
        self.simulator_params = simulator_params
        self.networkx_graph = networkx_graph
        self.nx_graph, self.dataset, self.aware_simuator, self.progress_placer = None, None, None, None

        self.comfirm_params()

    def comfirm_params(self):
        if self.config_params["eval_freq"] % 10 == 0:
            raise ValueError("Eval freq cannot be divisible by 10")

    def startup_strategy(self):
        """
        startup strategy
        """
        # pylint: disable=W0718
        try:
            self.nx_graph = copy.deepcopy(self.networkx_graph.get_nx_graph())
            self.aware_simuator = AwareSimulator(self.config_params, self.nx_graph)
            self.progress_placer = ProgressivePlacer(
                self.nx_graph, self.config_params, self.simulator_params, self.aware_simuator
            )
            self.progress_placer.place()
            return True
        except Exception as e:  # catch all exception
            traceback.print_exc(e)
            return False
        # pylint: enable=W0718

    def get_best_placement(self):
        return self.progress_placer.best_placement
