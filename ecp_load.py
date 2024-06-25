from utils import *
from loss import *

if __name__ == "__main__":
    device = get_device()
    print('Using Device is: {}'.format(device))
    last_epoch = None
    best_acc = 100
    input_data = 'imagenet' # OR 'imagenetv2'
    num_classes = 1000
    n_calib = 15000 # OR 3000 for imagenetv2
    alpha_val = 0.1 # user-specified coverage error level

    ########## Data
    print('==> Preparing data..')
    loaders = data_choice_val(dataset=input_data, dl=False, num_calib=n_calib)
    
    # Get the model (change it based on your desired architecture. Only one pretrained model can be used at a time.)
    print('==> Building model..')
    
    # net = torchvision.models.resnext101_32x8d(weights='IMAGENET1K_V1', progress=True)
    net = torchvision.models.resnet152(weights='IMAGENET1K_V1', progress=True)
    # net = torchvision.models.resnet101(weights='IMAGENET1K_V1', progress=True)
    # net = torchvision.models.resnet50(weights='IMAGENET1K_V1', progress=True)
    # net = torchvision.models.resnet18(weights='IMAGENET1K_V1', progress=True)
    # net = torchvision.models.densenet161(weights='IMAGENET1K_V1', progress=True)
    # net = torchvision.models.vgg16(weights='IMAGENET1K_V1', progress=True)
    # net = torchvision.models.shufflenet_v2_x1_0(weights='IMAGENET1K_V1', progress=True)
    # net = torchvision.models.inception_v3(weights='IMAGENET1K_V1', progress=True)
    
    net = net.to(device)

    ######## Loss and Hyperparameters 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, 
    #                        weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    logits, labels, acc, temp_val = test_cp(last_epoch, num_classes, loaders['loader_tup'][0], loaders['val_same'], net, 
                                            best_acc, criterion, device, data_name=input_data)
    
    tup_dict = {'base': None, 'aps': None, 'raps': None, 'ecp': None}
    loop = True
    while loop:
        kreg = int(input('\nEnter k_reg for RAPS (Enter -1 to Stop the Process): '))
        if kreg == -1:
            loop = False
        else:
            lamdareg = float(input('\nEnter lam_reg for RAPS: '))
            
            p_size, size_tup, std_tup, cov_tup, true_rank, cal_num, alpha, strat_tup, sscv, sa_trade = cp_generator(logits, labels, 
            temp_scale=temp_val, n=n_calib, alpha=alpha_val, method='ecp', lam_reg=0)
            print('Mean prediction set size for {} is: {}'.format('ECP', np.round(p_size.mean(),3)))
            print()
            print('Size-stratified Coverage Violation in {} is: {}'.format('ECP', sscv))
            print()
            print('Size-Adaptivity Trade-off in {} is: {}'.format('ECP', sa_trade))
            print()
            j = 0
            for i, std_element in enumerate(std_tup):
                print('Mean Set Size (Difficulty) - Group {} in {} is: {}'.format(i+1, 'ECP', size_tup[i]))
                print('Standard Deviation (Difficulty) - Group {} in {} is: {}'.format(i+1, 'ECP', std_element))
                print('Counts of instances (Difficulty) - Group {} in {} is: {}'.format(i+1, 'ECP', cov_tup[1][i]))
                print('Marginal Coverage (Difficulty) - Group {} in {} is: {}'.format(i+1, 'ECP', cov_tup[0][i]))
                print('Counts of Size Range - Group {} in {} is: {}'.format(i+1, 'ECP', strat_tup[1][i]))
                if strat_tup[1][i] == 0: 
                    print('Size Range Coverage - Group {} in {} is: {}'.format(i+1, 'ECP', '---'))
                else:
                    print('Size Range Coverage - Group {} in {} is: {}'.format(i+1, 'ECP', strat_tup[0][j]))
                    j += 1
                print()
            tup_dict['ecp'] = size_tup
            
            p_size, size_tup, std_tup, cov_tup, true_rank, cal_num, alpha, strat_tup, sscv, sa_trade = cp_generator(logits, labels, 
            temp_scale=temp_val, n=n_calib, alpha=alpha_val, method='naive', lam_reg=0)
            print('Mean prediction set size for {} is: {}'.format('Base', np.round(p_size.mean(),3)))
            print()
            print('Size-stratified Coverage Violation in {} is: {}'.format('Base', sscv))
            print()
            print('Size-Adaptivity Trade-off in {} is: {}'.format('Base', sa_trade))
            print()
            j = 0
            for i, std_element in enumerate(std_tup):
                print('Mean Set Size (Difficulty) - Group {} in {} is: {}'.format(i+1, 'Base', size_tup[i]))
                print('Standard Deviation (Difficulty) - Group {} in {} is: {}'.format(i+1, 'Base', std_element))
                print('Counts of instances (Difficulty) - Group {} in {} is: {}'.format(i+1, 'Base', cov_tup[1][i]))
                print('Marginal Coverage (Difficulty) - Group {} in {} is: {}'.format(i+1, 'Base', cov_tup[0][i]))
                print('Counts of Size Range - Group {} in {} is: {}'.format(i+1, 'Base', strat_tup[1][i]))
                if strat_tup[1][i] == 0: 
                    print('Size Range Coverage - Group {} in {} is: {}'.format(i+1, 'Base', '---'))
                else:
                    print('Size Range Coverage - Group {} in {} is: {}'.format(i+1, 'Base', strat_tup[0][j]))
                    j += 1
                print()
            tup_dict['base'] = size_tup
            
            p_size, size_tup, std_tup, cov_tup, true_rank, cal_num, alpha, strat_tup, sscv, sa_trade = cp_generator(logits, labels, 
            temp_scale=temp_val, n=n_calib, alpha=alpha_val, method='aps', lam_reg=0)
            print('Mean prediction set size for {} is: {}'.format('APS', np.round(p_size.mean(),3)))
            print()
            print('Size-stratified Coverage Violation in {} is: {}'.format('APS', sscv))
            print()
            print('Size-Adaptivity Trade-off in {} is: {}'.format('APS', sa_trade))
            print()
            j = 0
            for i, std_element in enumerate(std_tup):
                print('Mean Set Size (Difficulty) - Group {} in {} is: {}'.format(i+1, 'APS', size_tup[i]))
                print('Standard Deviation (Difficulty) - Group {} in {} is: {}'.format(i+1, 'APS', std_element))
                print('Counts of instances (Difficulty) - Group {} in {} is: {}'.format(i+1, 'APS', cov_tup[1][i]))
                print('Marginal Coverage (Difficulty) - Group {} in {} is: {}'.format(i+1, 'APS', cov_tup[0][i]))
                print('Counts of Size Range - Group {} in {} is: {}'.format(i+1, 'APS', strat_tup[1][i]))
                if strat_tup[1][i] == 0: 
                    print('Size Range Coverage - Group {} in {} is: {}'.format(i+1, 'APS', '---'))
                else:
                    print('Size Range Coverage - Group {} in {} is: {}'.format(i+1, 'APS', strat_tup[0][j]))
                    j += 1
                print()
            tup_dict['aps'] = size_tup
            
            p_size, size_tup, std_tup, cov_tup, true_rank, cal_num, alpha, strat_tup, sscv, sa_trade = cp_generator(logits, labels, 
            temp_scale=temp_val, n=n_calib, alpha=alpha_val, method='raps', lam_reg=lamdareg, k_reg=kreg)
            print('Mean prediction set size for {} is: {}'.format('RAPS', np.round(p_size.mean(),3)))
            print()
            print('Size-stratified Coverage Violation in {} is: {}'.format('RAPS', sscv))
            print()
            print('Size-Adaptivity Trade-off in {} is: {}'.format('RAPS', sa_trade))
            print()
            j = 0
            for i, std_element in enumerate(std_tup):
                print('Mean Set Size (Difficulty) - Group {} in {} is: {}'.format(i+1, 'RAPS', size_tup[i]))
                print('Standard Deviation (Difficulty) - Group {} in {} is: {}'.format(i+1, 'RAPS', std_element))
                print('Counts of instances (Difficulty) - Group {} in {} is: {}'.format(i+1, 'RAPS', cov_tup[1][i]))
                print('Marginal Coverage (Difficulty) - Group {} in {} is: {}'.format(i+1, 'RAPS', cov_tup[0][i]))
                print('Counts of Size Range - Group {} in {} is: {}'.format(i+1, 'RAPS', strat_tup[1][i]))
                if strat_tup[1][i] == 0: 
                    print('Size Range Coverage - Group {} in {} is: {}'.format(i+1, 'RAPS', '---'))
                else:
                    print('Size Range Coverage - Group {} in {} is: {}'.format(i+1, 'RAPS', strat_tup[0][j]))
                    j += 1
                print()
            tup_dict['raps'] = size_tup
            
            p_size, size_tup, std_tup, cov_tup, true_rank, cal_num, alpha, strat_tup, sscv, sa_trade = cp_generator(logits, labels, 
            temp_scale=temp_val, n=n_calib, alpha=alpha_val, method='lac', lam_reg=0)
            print('Mean prediction set size for {} is: {}'.format('LAS', np.round(p_size.mean(),3)))
            print()
            print('Size-stratified Coverage Violation in {} is: {}'.format('LAS', sscv))
            print()
            print('Size-Adaptivity Trade-off in {} is: {}'.format('LAS', sa_trade))
            print()
            j = 0
            for i, std_element in enumerate(std_tup):
                print('Mean Set Size (Difficulty) - Group {} in {} is: {}'.format(i+1, 'LAS', size_tup[i]))
                print('Standard Deviation (Difficulty) - Group {} in {} is: {}'.format(i+1, 'LAS', std_element))
                print('Counts of instances (Difficulty) - Group {} in {} is: {}'.format(i+1, 'LAS', cov_tup[1][i]))
                print('Marginal Coverage (Difficulty) - Group {} in {} is: {}'.format(i+1, 'LAS', cov_tup[0][i]))
                print('Counts of Size Range - Group {} in {} is: {}'.format(i+1, 'LAS', strat_tup[1][i]))
                if strat_tup[1][i] == 0: 
                    print('Size Range Coverage - Group {} in {} is: {}'.format(i+1, 'LAS', '---'))
                else:
                    print('Size Range Coverage - Group {} in {} is: {}'.format(i+1, 'LAS', strat_tup[0][j]))
                    j += 1
                print()
            
            
            adapt_plot(tup_dict)

    
    