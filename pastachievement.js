import nj from "numjs";
import tf from "@tensorflow/tfjs";

import tfn from "@tensorflow/tfjs-node";
import fetch from "node-fetch";
global.fetch = fetch;

const input_arr = nj.ones([1, 500, 500, 1]);
// const handler = tfn.io.fileSystem("./model/model.json");
// tf.loadLayersModel(handler).then(function (model) {
//   window.model = model;
// });

const handler = tfn.io.fileSystem("./model/model.json");
tf.loadLayersModel(handler).then(function (model) {
  console.log("model is: ", model);
});

// async function loadModel() {
//   try {
//     const x = tf.input({ shape: [32] });
//     const y = tf.layers.dense({ units: 3, activation: "softmax" }).apply(x);
//     const model = tf.model({ inputs: x, outputs: y });
//     await model.predict(tf.ones([2, 32])).print();
//     const json_config = model.toJSON();
//     // console.log(json_config);
//     const new_model = tf.models.modelFromJSON(json_config);

//     // const handler = tfn.io.fileSystem("./model/model.json");
//     // const model = await tf.loadLayersModel(handler);
//     // console.log("Model loaded");
//     // console.log(model, new_model);
//     return new_model;
//   } catch (err) {
//     console.error(err);
//   }
// }

// const model = loadModel();

var predict = async function (input) {
  try {
    if (model) {
      model
        .predict([input_arr])
        .array()
        .then(function (scores) {
          scores = scores[0];
          predicted = scores.indexOf(Math.max(...scores));
          $("#number").html(predicted);
        });
    } else {
      // The model takes a bit to load,
      // if we are too fast, wait
      setTimeout(function () {
        predict(input);
      }, 50);
    }
  } catch (error) {
    console.error(error);
  }
};

// var predict = function (input) {
//     if (window.model) {
//         window.model.predict([tf.tensor(input)
//             .reshape([1, 28, 28, 1])])
//             .array().then(function (scores) {
//                 scores = scores[0];
//                 predicted = scores
//                     .indexOf(Math.max(...scores));
//                 $('#number').html(predicted);
//             });
//     } else {

//         // The model takes a bit to load,
//         // if we are too fast, wait
//         setTimeout(function () { predict(input) }, 50);
//     }
// }
