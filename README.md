# CMPM202-Project2

## Part 1 : Training the Magenta Drum RNN to Generate MIDI Drum Patterns
https://github.com/tensorflow/magenta/tree/master/magenta/models/drums_rnn

### Step 1 : Find a dataset

The Drum Percussion Midi Archive (800k) -- 890.9 MB Compressed || 3.36GB Uncompressed
https://www.reddit.com/r/WeAreTheMusicMakers/comments/3anwu8/the_drum_percussion_midi_archive_800k/

### Step 2 : Prepare the data

Magenta has a built in script for concatenating midi files in a directory into a single file (TFRecord).
Install Magenta using:

```
pip install magenta-gpu
```

To concatenate, use the script 'convert_dir_to_note_sequences"

```
INPUT_DIRECTORY=<folder containing MIDI and/or MusicXML files. can have child folders.>
SEQUENCES_TFRECORD=/tmp/notesequences.tfrecord #this is where the TFRecord file will be saved

convert_dir_to_note_sequences \
  --input_dir=$INPUT_DIRECTORY \
  --output_file=$SEQUENCES_TFRECORD \dir
  --recursive
```

### Step 3 : Create training and evaluation datasets 

```
drums_rnn_create_dataset \
--config=<one of 'one_drum' or 'drum_kit'> \ # I selected 'drum_kit'
--input=/tmp/notesequences.tfrecord \
--output_dir=/tmp/drums_rnn/sequence_examples \
--eval_ratio=0.10 # 10% of data is saved for evaluation
```
### Step 4 : Train the RNN

```
drums_rnn_train \
--config=drum_kit \
--run_dir=/tmp/drums_rnn/logdir/run1 \
--sequence_example_file=/tmp/drums_rnn/sequence_examples/training_drum_tracks.tfrecord \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--num_training_steps=20000
```
### Step 5 : Generate drum tracks
```
drums_rnn_generate \
--config=drum_kit \
--run_dir=/tmp/drums_rnn/logdir/run1 \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--output_dir=/tmp/drums_rnn/generated \
--num_outputs=10 \ # number of files that are generated
--num_steps=128 \ # number of steps -- each step is a 1/16 note
--primer_drums="[(36,)]" # give starting sequence of notes; (36,) is a single bass drum hit
```

## Part 2 : Transform audio using an autoencoder NN
Implementation of https://github.com/DmitryUlyanov/neural-style-audio-tf
Teach a NN the style of one audio file and transfer it to another.
This model has its limitations:
- Can only train on one Style example file --> will either move to a different model or modify this one
- Content / Style examples must be the same size

### Step 1 : Create audio file examples

I created two example drum tracks in Logic Pro X using MIDI generated in Part 1 of this project:
- One version without any filters / effects
- One version with distortion / compression / ...

Since the size between the two files needs to match, use the ffmpeg package to create shorter segments.

To cut each sound file down to the first 10 seconds (which I use in this example):
```
ffmpeg -i yourfile.mp3 -ss 00:00:00 -t 10 yourfile_10s.mp3
```

To cut each sound file into many 10 second chunks (which I would use if I want more robust training):
```
ffmpeg -i fx_off.wav -f segment -segment_time 10 -c copy fx_off_out%03d.wav
```

### Step 2 : 'Digest' data into TF friendly format

Tensorflow cannot natively handle .wav files, so they must be converted. The ```librosa``` package for Python has functions for reading audio files and running the Short-Time Fourier Transform, returning a matrix D, where np.abs(D[f,t]) is the magnitude of frequency bin f at frame t. This information can be used to generate a spectrogram. In this example, the function ```read_audio_spectrum()``` is defined to handle this transformation.


```
N_FFT = 2048 # length of the signal you want to calculate the Fourier transform
def read_audio_spectum(filename):
    x, fs = librosa.load(filename) # x = the input signal (audio time series); fs = sampling rate
    S = librosa.stft(x, N_FFT) # short-time fourier transform
    p = np.angle(S) # the phase information
    
    S = np.log1p(np.abs(S[:,:]))  
  return S, fs
```

Read in audio spectrum data
```
a_content, fs = read_audio_spectum(CONTENT_FILENAME)
a_style, fs = read_audio_spectum(STYLE_FILENAME)

N_SAMPLES = a_content.shape[1]
N_CHANNELS = a_content.shape[0]
a_style = a_style[:N_CHANNELS, :N_SAMPLES]
```

Generate spectrograms
```
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Content')
plt.imshow(a_content[:,:])
plt.subplot(1, 2, 2)
plt.title('Style')
plt.imshow(a_style[:,:])
plt.show()
```

### Step 3 : Train network

```
N_FILTERS = 4096

a_content_tf = np.ascontiguousarray(a_content.T[None,None,:,:])
a_style_tf = np.ascontiguousarray(a_style.T[None,None,:,:])

# filter shape is "[filter_height, filter_width, in_channels, out_channels]"
std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 11))
kernel = np.random.randn(1, 11, N_CHANNELS, N_FILTERS)*std
    
g = tf.Graph()
with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
    # data shape is "[batch, in_height, in_width, in_channels]",
    x = tf.placeholder('float32', [1,1,N_SAMPLES,N_CHANNELS], name="x")

    kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
    conv = tf.nn.conv2d(
        x,
        kernel_tf,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
    
    net = tf.nn.relu(conv)

    content_features = net.eval(feed_dict={x: a_content_tf})
    style_features = net.eval(feed_dict={x: a_style_tf})
    
    features = np.reshape(style_features, (-1, N_FILTERS))
    style_gram = np.matmul(features.T, features) / N_SAMPLES
    
from sys import stderr

ALPHA= 4
learning_rate= 1e-3
iterations = 100

result = None
with tf.Graph().as_default():
    x = tf.Variable(np.random.randn(1,1,N_SAMPLES,N_CHANNELS).astype(np.float32)*1e-3, name="x")

    kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
    conv = tf.nn.conv2d(
        x,
        kernel_tf,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
    
    
    net = tf.nn.relu(conv)

    content_loss = ALPHA * 2 * tf.nn.l2_loss(
            net - content_features)

    style_loss = 0

    _, height, width, number = map(lambda i: i.value, net.get_shape())

    size = height * width * number
    feats = tf.reshape(net, (-1, number))
    gram = tf.matmul(tf.transpose(feats), feats)  / N_SAMPLES
    style_loss = 2 * tf.nn.l2_loss(gram - style_gram)

     # Overall loss
    loss = content_loss + style_loss

    opt = tf.contrib.opt.ScipyOptimizerInterface(
          loss, method='L-BFGS-B', options={'maxiter': 300})
        
    # Optimization
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
       
        print('Started optimization.')
        opt.minimize(sess)
    
        print('Final loss:', loss.eval())
        result = x.eval()
 
```

### Step 4 : Generate output

Applies the learned Style onto the Content file that was given at the beginning
```
a = np.zeros_like(a_content)
a[:N_CHANNELS,:] = np.exp(result[0,0].T) - 1

# This code is supposed to do phase reconstruction
p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
for i in range(500):
    S = a * np.exp(1j*p)
    x = librosa.istft(S)
    p = np.angle(librosa.stft(x, N_FFT))

OUTPUT_FILENAME = 'out2.wav'
librosa.output.write_wav(OUTPUT_FILENAME, x, fs)
```

### Results

Even with limited training examples this model, to an extent, is able to learn and transfer an audio style.
Could be used for decompressing / denoising an audio signal
