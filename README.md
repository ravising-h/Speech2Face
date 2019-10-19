## Speech2Face
### This repository has all the codes of my implementation of Speech to face.
![Link to The Paper](https://arxiv.org/pdf/1905.09773.pdf)
![](https://thumbs-prod.si-cdn.com/LvZAAPgi3v9zreVi5wB8Y3jGWN4=/800x600/filters:no_upscale()/https://public-media.si-cdn.com/filer/22/b3/22b3449f-8948-44b9-967c-10911b729494/ahr0cdovl3d3dy5saxzlc2npzw5jzs5jb20vaw1hz2vzl2kvmdawlzewni8wmjgvb3jpz2luywwvywktahvtyw4tdm9py2utznjvbs1mywnl.jpeg)

#### Requirements

* Python 3.5 or above
* Keras
* TensorFlow
* Librosa
* keras_vggface
* opencv
* Dlib



#### Speech processing

Done:
* Loaded the wav file
* Did STFT of the same with hopplenght = 10ms and hann_window = 10 mm
* Power Law compression Sig(Spectrum)|Spectrum|^0.3

#### Image Processing

Done:
* loaded the dataset (according to names used in speech processing)
* detected Face Using `DLib CNN`
* Calculated the Face Feature using VGG16 Model
* Saved the croped faces and feature for Face Decoder


### MODEL

Done:
1) Build and Trained Voice Encoder Model having 0.715 mean absolute  loss


Model is Implemented in TensorFlow with Adam Optimizer where value of Beta 1 is 0.5 and Value Of Beta 2 is 0.9899.

![](https://www.i-programmer.info/images/stories/News/2019/june/A/voice2face.jpg)

**TO DO:**

1) Build Face Decoder Model
2) Build Speech to Cartoon
