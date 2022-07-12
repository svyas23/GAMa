
# trial 4:_Combined: centering + adam + single gpu + aug + hn+ sat transformer

#weights = 'save/SupCon/bdd_vgl_models/SupCon_bdd_vgl_resnet18_lr_8e-05_decay_0.0001_bsz_64_temp_0.1_trial_4_dim512/ckpt_epoch_400.pth'
weights = './GAMa_Net/save/gama_ckpt_epoch_400.pth'
# all distance corrected: (array([0.15311653, 0.27155507, 0.33798526, 0.91941553]), 97.12469194529415) # confirm (array([0.15311653, 0.27155507, 0.33798526]), 97.12469194529415) #initial saves: (array([0.15311653, 0.27155507, 0.33798526]), 97.12469194529415)

#metric threshold: top1; 0.1, 0.2, 0.5, 1.0 mile: (array([0.19618352, 0.22998982, 0.28746138, 0.36132256]), 97.12469194529415)











