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
    
    model_dict = {'resnext101': models.resnext101_32x8d(weights='IMAGENET1K_V1', progress=True),
                  'resnet152': models.resnet152(weights='IMAGENET1K_V1', progress=True),
                  'resnet101': models.resnet101(weights='IMAGENET1K_V1', progress=True),
                  'resnet50': models.resnet50(weights='IMAGENET1K_V1', progress=True),
                  'resnet18': models.resnet18(weights='IMAGENET1K_V1', progress=True),
                  'densenet161': models.densenet161(weights='IMAGENET1K_V1', progress=True),
                  'vgg16': models.vgg16(weights='IMAGENET1K_V1', progress=True),
                  'shufflenet': models.shufflenet_v2_x1_0(weights='IMAGENET1K_V1', progress=True),
                  'inception': models.inception_v3(weights='IMAGENET1K_V1', progress=True)
                  }
    
    for name, m in model_dict.items():
        print()
        print('=========================================')
        print('***Using Pretrained Model: {}'.format(name))
        print()
        
        net = m
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
        
        lambda_set = [0.00001, 0.0001, 0.0008, 0.001, 0.0015, 0.002, 0.01, 0.1, 1]
        sscv_set, size_set, sat_set = [], [], []
         
        p_size, size_tup, std_tup, cov_tup, true_rank, cal_num, alpha, strat_tup, sscv, sa_trade = cp_generator(logits, labels, 
        temp_scale=temp_val, n=n_calib, alpha=alpha_val, method='ecp', lam_reg=0, verbose=False)
        print('Mean prediction set size for {} is: {}'.format('ECP', np.round(p_size.mean(),3)))
        print('SSCV in {} is: {}'.format('ECP', sscv))
        print('Size-Adaptivity Trade-off in {} is: {}'.format('ECP', sa_trade))
        print()
        
        p_size, size_tup, std_tup, cov_tup, true_rank, cal_num, alpha, strat_tup, sscv, sa_trade = cp_generator(logits, labels, 
        temp_scale=temp_val, n=n_calib, alpha=alpha_val, method='aps', lam_reg=0, verbose=False)
        print('Mean prediction set size for {} is: {}'.format('APS', np.round(p_size.mean(),3)))
        print('SSCV in {} is: {}'.format('APS', sscv))
        print('Size-Adaptivity Trade-off in {} is: {}'.format('APS', sa_trade))
        print()
        
        p_size, size_tup, std_tup, cov_tup, true_rank, cal_num, alpha, strat_tup, sscv, sa_trade = cp_generator(logits, labels, 
        temp_scale=temp_val, n=n_calib, alpha=alpha_val, method='lac', lam_reg=0, verbose=False)
        print('Mean prediction set size for {} is: {}'.format('LAS', np.round(p_size.mean(),3)))
        print('SSCV in {} is: {}'.format('LAS', sscv))
        print('Size-Adaptivity Trade-off in {} is: {}'.format('LAS', sa_trade))
        print()

        for lam in tqdm(lambda_set):
            p_size, size_tup, std_tup, cov_tup, true_rank, cal_num, alpha, strat_tup, sscv, sa_trade = cp_generator(logits, labels, 
            temp_scale=temp_val, n=n_calib, alpha=alpha_val, method='raps', lam_reg=lam, verbose=False)
            # print('Mean prediction set size in {} for Lambda={} is: {}'.format('RAPS', lam, np.round(p_size.mean(),3)))
            # print('SSCV in {} for Lambda={} is: {}'.format('RAPS', lam, sscv))
            # print('Size-Adaptivity Trade-off in {} for Lambda={} is: {}'.format('RAPS', lam, sa_trade))
            # print()
            sscv_set.append(sscv)
            size_set.append(np.round(p_size.mean(),3))
            sat_set.append(sa_trade)
        
        sscv_min_arg = np.argmin(sscv_set)
        print('*** Minimum SSCV for RAPS is {} for Lambda={}:\nMean Set Size: {}\nSAT Score: {}'.format(np.min(sscv_set), 
        lambda_set[sscv_min_arg], size_set[sscv_min_arg], sat_set[sscv_min_arg]))
    
    

