importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");

//importScripts("./opencv.js/opencv.js");


onmessage = async(event) => {
    const input = event.data;
    console.log(input);
    const output = await run_model(input);
    postMessage(output);
}

async function run_model(input) {
    //importScripts("./source/ort-wasm-simd.wasm");
    const model = await ort.InferenceSession.create("/source/license_plate_detector.onnx");
    input = new ort.Tensor(Float32Array.from(input),[1, 3, 640, 640]);
    const outputs = await model.run({images:input});
    return outputs["output0"].data;
}