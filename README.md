# image-classification
This work was my Bachelor thesis in 2020. The idea is to compare the newly proposed Bagnet's performance on image classificaiton with classical networks like VGG and Resnet.
I choose both ILSVRC2012 _val and Caltech101 dataset. 

For Caltech101 set, I used both Imagenet pretrained model (transfer learning) and untrained model, change the last layeroutput of those nets, train them on caltech101 dataset 
with data augmentation and then do the validation.

For Imagenet set, I used pretrained models.
My running results can be seen in the text "short summary".
And unfortunately, all the comments were written in Chinese :)

The whole work was based on https://github.com/wielandbrendel/bag-of-local-features-models

It also includes a bagnet visualization part "heatmap.py", which refers to https://blog.csdn.net/winycg/article/details/101269632
