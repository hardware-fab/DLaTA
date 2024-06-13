"""
Authors : 
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari,
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

import CNN.modules as modules

def build_module(module_config, gpu):
    module_name = module_config['module']['name']
    module_config = module_config['module']['config']
    module_class = getattr(modules, module_name)
    module = module_class(module_config, gpu)
    return module