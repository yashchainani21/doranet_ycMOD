import sys
from pathlib import Path

# Add the local doranet_ycMOD to the path (before pip-installed doranet)
DORANET_YCMOD_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(DORANET_YCMOD_PATH))

import doranet.modules.enzymatic as enzymatic
import doranet.modules.synthetic as synthetic
import doranet.modules.post_processing as post_processing

user_starters = {'C1=C(C(=CC=O)OC1=O)N'}

user_helpers = {'O','O=O','[H][H]','O=C=O','C=O','[C-]#[O+]','Br','[Br][Br]','CO',
                'C=C','O=S(O)O','N','O=S(=O)(O)O','O=NO','N#N','O=[N+]([O-])O','NO',
                'C#N','S','O=S=O','N#CO'}

user_target = {'OC1=CC=CC=C1'}        

job_name = "test_enzymatic"

forward_network = enzymatic.generate_network(
    job_name = job_name,
    starters = user_starters,
    gen = 1,
    direction = "forward",
    ruleset = "JN3604IMT")

# forward_network = synthetic.generate_network(
#     job_name = job_name,
#     starters = user_starters,
#     helpers = user_helpers,
#     gen = 1,
#     direction = "forward",
#     ruleset = "JN3604IMT"
# )

for mol in forward_network.mols:
    print(mol.uid)

post_processing.one_step(
    networks = {
        forward_network,
        },
    total_generations = 1,
    starters = user_starters,
    helpers = user_helpers,
    target = user_target,
    job_name = job_name,
    )
