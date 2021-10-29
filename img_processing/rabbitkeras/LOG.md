# Trace major issue met and the quick solutions

* Issue:
1. Same model, ipython will get lower loss rather than python
>> This is not accurate description, the truth is following 2 kind loss
>> This is caused by random initialization, but simple change to another GPU works better than restart
- Loss not go down
Epoch 1/50
802/802 [==============================] - 46s 58ms/sample - loss: 0.0973 - val_loss: 0.0977
Epoch 2/50
802/802 [==============================] - 31s 39ms/sample - loss: 0.0973 - val_loss: 0.0977
- Loss go down well
Epoch 1/10
802/802 [==============================] - 36s 45ms/sample - loss: 0.0778 - val_loss: 0.0757
Epoch 2/10
802/802 [==============================] - 32s 40ms/sample - loss: 0.0737 - val_loss: 0.0751

2. Need a better accuracy now psnr has been done, need ssim.
* psnr in training
- Epoch 1/50
200/200 [==============================] - 14s 70ms/sample - loss: 0.0810 - acc: 0.3370 - psnr: 20.8146 - val_loss: 0.0755 - val_acc: 0.3411 - val_psnr: 20.7281
- Epoch 2/50
200/200 [==============================] - 8s 41ms/sample - loss: 0.0731 - acc: 0.3374 - psnr: 21.5187 - val_loss: 0.0754 - val_acc: 0.3412 - val_psnr: 21.0877
- Epoch 3/50
200/200 [==============================] - 11s 55ms/sample - loss: 0.0726 - acc: 0.3374 - psnr: 21.5769 - val_loss: 0.0743 - val_acc: 0.3412 - val_psnr: 20.8722
- ...
- Epoch 49/50
200/200 [==============================] - 11s 55ms/sample - loss: 0.0691 - acc: 0.3375 - psnr: 21.8188 - val_loss: 0.0733 - val_acc: 0.3412 - val_psnr: 21.1993
- Epoch 50/50
200/200 [==============================] - 8s 41ms/sample - loss: 0.0690 - acc: 0.3375 - psnr: 21.8253 - val_loss: 0.0732 - val_acc: 0.3412 - val_psnr: 21.1551


