#include <AL/al.h>    // OpenAL header files
#include <AL/alc.h>
#include <iostream>
#include <list>
#include <fann.h>

using std::list;

#define SZ 16
#define FREQ 44100 / 2  // Sample rate
#define CAP_SIZE 2048 // How much to capture at a time (affects latency)

const unsigned int num_input = FREQ*SZ;
    const unsigned int num_output = FREQ;
    const unsigned int num_layers = 3;
    const unsigned int num_neurons_hidden = 256;
    const float desired_error = (const float) 0.000001;
    const unsigned int max_epochs = 1;
    const unsigned int epochs_between_reports = 100;

    struct fann *ann = fann_create_standard(num_layers, num_input,
        num_neurons_hidden, num_output);

 

  //  fann_train_on_file(ann, "xor.data", max_epochs, epochs_between_reports, desired_error);

 //   fann_save(ann, "xor_float.net");



int iterations = 0;
int limit = 128;
void processBuffer(short* buffer,int len) {
    fann_type buf[len];
    for (int i = 0; i < len; i++) {
        buf[i] = buffer[i];
    }
    if (iterations < limit) {
        std::cout << "Started training " << iterations << std::endl <<std::flush;
        fann_train(ann, buf, buf);
        std::cout << "Ended training" << std::endl << std::flush;
    }
    

    if (iterations == limit) {
        fann_save(ann, "xor_float.net");
    }
 
    
}

int main(int argC,char* argV[])
{
       fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

    list<ALuint> bufferQueue; // A quick and dirty queue of buffer objects

    ALenum errorCode=0;
    ALuint helloBuffer[16], helloSource[1];
    ALCdevice* audioDevice = alcOpenDevice(NULL); // Request default audio device
    errorCode = alcGetError(audioDevice);
    ALCcontext* audioContext = alcCreateContext(audioDevice,NULL); // Create the audio context
    alcMakeContextCurrent(audioContext);
    errorCode = alcGetError(audioDevice);
    // Request the default capture device with a half-second buffer
    ALCdevice* inputDevice = alcCaptureOpenDevice(NULL,FREQ,AL_FORMAT_MONO16,FREQ/2);
    errorCode = alcGetError(inputDevice);
    alcCaptureStart(inputDevice); // Begin capturing
    errorCode = alcGetError(inputDevice);

    alGenBuffers(16,&helloBuffer[0]); // Create some buffer-objects
    errorCode = alGetError();

    // Queue our buffers onto an STL list
    for (int ii=0;ii<16;++ii) {
        bufferQueue.push_back(helloBuffer[ii]);
    }

  alGenSources (1, &helloSource[0]); // Create a sound source
    errorCode = alGetError();

    short buffer[FREQ*SZ]; // A buffer to hold captured audio
    ALCint samplesIn=0;  // How many samples are captured
    ALint availBuffers=0; // Buffers to be recovered
    ALuint myBuff; // The buffer we're using
    ALuint buffHolder[16]; // An array to hold catch the unqueued buffers
    bool done = false;
    while (!done) { // Main loop
        // Poll for recoverable buffers
        alGetSourcei(helloSource[0],AL_BUFFERS_PROCESSED,&availBuffers);
        if (availBuffers>0) {
            alSourceUnqueueBuffers(helloSource[0],availBuffers,buffHolder);
            for (int ii=0;ii<availBuffers;++ii) {
                // Push the recovered buffers back on the queue
                bufferQueue.push_back(buffHolder[ii]);
            }
        }
        // Poll for captured audio
        alcGetIntegerv(inputDevice,ALC_CAPTURE_SAMPLES,1,&samplesIn);
        if (samplesIn>CAP_SIZE) {
            // Grab the sound
            alcCaptureSamples(inputDevice,buffer,CAP_SIZE);

            //***** Process/filter captured data here *****//
            //for (int ii=0;ii<CAP_SIZE;++ii) {
            //  buffer[ii]*=0.1; // Make it quieter
            //}

            // Stuff the captured data in a buffer-object
            if (!bufferQueue.empty()) { // We just drop the data if no buffers are available
                myBuff = bufferQueue.front(); bufferQueue.pop_front();
                alBufferData(myBuff,AL_FORMAT_MONO16,buffer,CAP_SIZE*sizeof(short),FREQ);

                // Queue the buffer
                alSourceQueueBuffers(helloSource[0],1,&myBuff);

                // Restart the source if needed
                // (if we take too long and the queue dries up,
                //  the source stops playing).
                ALint sState=0;
                alGetSourcei(helloSource[0],AL_SOURCE_STATE,&sState);
                 
                
           

        }
           

    }
                if (iterations > limit) {
                   fann_type buf[FREQ*SZ];
                 
                    fann_type *calc_out = fann_run(ann, buf);

                        for (int i = 0; i < FREQ*SZ; i++) {
                            buf[i] = buffer[i];
                        }
                  alSourcePlay(calc_out[0]); 
                } else {

                  processBuffer(buffer, FREQ*SZ);
                    
                     iterations++;   
            }  
                
        
    }
    // Stop capture
    alcCaptureStop(inputDevice);
    alcCaptureCloseDevice(inputDevice);

    // Stop the sources
    alSourceStopv(1,&helloSource[0]);
    for (int ii=0;ii<1;++ii) {
        alSourcei(helloSource[ii],AL_BUFFER,0);
    }
    // Clean-up 
    alDeleteSources(1,&helloSource[0]); 
    alDeleteBuffers(16,&helloBuffer[0]);
    errorCode = alGetError();
    alcMakeContextCurrent(NULL);
    errorCode = alGetError();
    alcDestroyContext(audioContext);
    alcCloseDevice(audioDevice);
    fann_destroy(ann);
    return 0;
}
