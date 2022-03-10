import nj from "numjs";
import tf from "@tensorflow/tfjs";

import tfn from "@tensorflow/tfjs-node";
import fetch from "node-fetch";
import { Image } from "image-js";
global.fetch = fetch;

const input_arr = nj.random([1, 500, 500, 1]).tolist();

const modelPath = "./model/model.json";

async function execute() {
  try {
    let image = await Image.load("./media/normal.jpg");
    image = image.resize({
      width: 500,
      height: 500,
    });
    let grey_image = image.grey();
    let converted_img = nj.array(grey_image.data);
    converted_img = converted_img.reshape([1, 500, 500, 1]);
    // console.log(converted_img[1][1]);
    // console.log(converted_img.shape);
    // input_arr = converted_img;
    // console.log(
    //   converted_img.slice([null, null], [null, null], [0], [0]).shape
    // );
    // console.log(converted_img)
    return grey_image;
  } catch (error) {
    console.error(error);
  }
}

// const input_arr = execute();

async function loadModel(modelPath) {
  try {
    const handler = tfn.io.fileSystem(modelPath);
    const model = await tf.loadLayersModel(handler);
    // tf.tensor(input_arr).print();
    // model.summary();

    const list = await input_arr;
    const ar2 = nj.images.read("./media/pneumonia.jpg");
    console.log(ar2.shape);
    const ar4 = nj.images.resize(ar2, 500, 500);
    const ar5 = nj.divide(ar4, 255);
    const ar3 = ar5.reshape([1, 500, 500, 1]).tolist();

    model.summary();
    const prediction = model.predict(tf.tensor(ar3));
    prediction.print();
    return model;
  } catch (error) {
    console.error(error);
  }
}

const new_model = loadModel(modelPath);
