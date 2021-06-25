var tf = require('@tensorflow/tfjs-node-gpu')
var RATE = 16000
var SZ = 12;


var model = tf.sequential();
var dropout = 0.002
model.add(tf.layers.dense({units: 12, inputShape: [(RATE/4) * SZ], activation: "linear" , dropout: dropout }));
model.add(tf.layers.dense({units: 512, activation: "linear", dropout: dropout }));
model.add(tf.layers.dense({units: 1024, activation: "linear", dropout: dropout }));
model.add(tf.layers.dense({units: 256, activation: "linear", dropout: dropout }));

model.add(tf.layers.dense({units: RATE/4, activation: "linear", dropout: dropout }));

model.compile({optimizer: tf.train.rmsprop(0.000015), loss: tf.losses.meanSquaredError , metrics: ['accuracy'] });

const Speaker = require('speaker');

// Create the Speaker instance
const speaker = new Speaker({
  channels: 1,          // 2 channels
  bitDepteh: 16,        // 16-bit samples
  sampleRate: RATE,     // 44,100 Hz sample rat
  signed: true,
  float: false
});

let Mic = require('node-microphone');
let mic = new Mic({
	device: 'default',
	rate: RATE,
	bitwidth: 16,
   	channels: 1,
    
});
let micStream = mic.startRecording();

var canTrain = true

var lT = Date.now()
var buf = new Buffer.allocUnsafe( (RATE*SZ)/4)
micStream.on('data', async (data) => {
	console.log(Date.now() - lT)
	lT = Date.now()
	if (data.length < RATE / 4) {
		data = Buffer.concat([data, new Buffer.alloc(RATE/4-data.length).fill(0)])
	} 
	if (data.length > RATE / 4) {
		data = data.slice(0, RATE / 4)
	}
	
		//var buffer = Buffer.concat(sampleArray)

	    
if (canTrain ) {
			canTrain = false
       			    
                var xs = tf.tensor2d(buf, [1, buf.length])
                var ys = tf.tensor2d(data, [1, data.length])
       			model.fit(xs, ys, { epochs: 1, validationData: [xs, ys], verbose: false } ).then(async () => { 
                    	canTrain = true;
                    	(async () => {
                    		//model.save('file://mymodel.json')         
                    	})()
                })
		}
		
		
	    
	//sampleArray.push(data);
	//
	// Get samples from zero to before final sample
	// write at end
	buf.copy(buf, 0, RATE/4, (RATE/4) * (SZ));
	data.copy(buf, RATE/4*(SZ), 0, RATE/4)
	
	var d = (await model.predict(tf.tensor2d(buf, [1, buf.length]) )).dataSync(); 		
	speaker.write(Buffer.from(Int16Array.from(d)));	
//	buf.write(data.toString('UTF-16LE'), (RATE * (SZ-1))/4,RATE/4, 'UTF-16LE');

})

var w = setInterval(async () => {


	


}, 128);
