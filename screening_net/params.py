

# 2DCNN ; 32 small aerial, small aerial vs larger aerial: Combined: centering + adam + single gpu + aug; NO hard negatives hn

#weights = './save/SupCon/bdd_vgl_models/SupCon_bdd_vgl_resnet18_lr_8e-05_decay_0.0001_bsz_8_temp_0.1_trial_2_dim512/ckpt_epoch_170.pth'#(array([0.07570876, 0.18460052, 0.26514175, 0.42139175, 0.78221649, 0.87242268, 0.95811856]), 102.5794098390592) # finetuning from170

# finetuning 2DCNN after 170 epoch ; 32 small aerial, small aerial vs larger aerial: Combined: centering + adam + single gpu + aug; NO hard negatives hn

#weights = './save/SupCon/bdd_vgl_models/SupCon_bdd_vgl_resnet18_lr_8e-05_decay_0.0001_bsz_8_temp_0.1_trial_4_finetuning_dim512/ckpt_epoch_150.pth' 
weights = './screening_net/save/finetuned_ckpt_epoch_150.pth'#(array([0.08215206, 0.20876289, 0.2931701 , 0.46262887, 0.83795103, 0.91527062, 0.97777062]), 86.29218896211421) # using for hier step

