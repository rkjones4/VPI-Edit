import os, sys
import torch
import utils
import json


# Each domain should be imported and returned, depending on the domain_name argument
def load_domain(args):    

    if args.domain_name == 'lay':
        from domains.layout import LAYOUT_DOMAIN
        return LAYOUT_DOMAIN()    
    
    elif args.domain_name == 'csg2d':
        from domains.csg2d import CSG2D_DOMAIN
        return CSG2D_DOMAIN()

    elif args.domain_name == 'csg3d':
        from domains.csg3d import CSG3D_DOMAIN
        return CSG3D_DOMAIN()
    
    else:
        assert False, f'bad domain name {args.domain_name}'
    
def main():
    main_args = utils.getArgs([
        ('-mm', '--main_mode', None, str), 
        ('-dn', '--domain_name', None, str), 
    ])
    
    domain = load_domain(main_args)

    if main_args.main_mode == 'os_pretrain':
        import os_pretrain
        return os_pretrain.pretrain(domain)

    elif main_args.main_mode == 'edit_pretrain':
        import edit_pretrain
        return edit_pretrain.pretrain(domain)
    
    elif main_args.main_mode == 'finetune':
        import joint_finetune
        return joint_finetune.fine_tune(domain)

    elif main_args.main_mode == 'eval':
        import joint_finetune
        return joint_finetune.eval(domain)
        
    else:
        assert False, f'bad main main {main_args.main_mode}'

            
if __name__ == '__main__':    
    main()

