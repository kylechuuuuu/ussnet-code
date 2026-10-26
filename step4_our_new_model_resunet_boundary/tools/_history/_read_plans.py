import sys, json
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

plans = json.load(open(sys.argv[1]))
pm = PlansManager(plans)
cm = pm.get_configuration('2d')
print('network_arch_class_name =', cm.network_arch_class_name)
print('network_arch_init_kwargs_req_import =', cm.network_arch_init_kwargs_req_import)
ak = cm.network_arch_init_kwargs
print('network_arch_init_kwargs =')
for k in sorted(ak.keys()):
    v = ak[k]
    print('   %-28s %s' % (k, v))
