import * as React from 'react';
import { useState, useEffect, useRef } from 'react';
import * as ReactDOM from 'react-dom';
import './opencv.scss';
const cv = require('opencv.js');
const Tesseract = require("tesseract.js");
import { Tensor, InferenceSession } from "onnxjs";
//import { Tensor, InferenceSession } from "onnxruntime-web";
//var fs = require('fs');
const ndarray = require('ndarray');
const ops = require('ndarray-ops');


export interface NumProps {
    number: number;
    canDraw:boolean;
    inputArr:number[];
    setInputArr:React.Dispatch<React.SetStateAction<number[]>>
}

export const Opencv: React.FC = () => {


    const worker = new Worker("/source/worker.js");
    let boxes : any = [];
    let interval;
    let busy = false;

    const [darkSetting, setDarkSetting] = useState<number>(85);

    const Plus =() =>{
        setDarkSetting(darkSetting+5)
    }
    const Minus =() =>{
        setDarkSetting(darkSetting-5)
    }
    

    const Detect = ()=>{
        const canvas = document.querySelector("#origin") as HTMLCanvasElement;
        const detectImg = document.querySelector("#detectImg") as HTMLImageElement;

        const canvas2 = document.querySelector("#originCanvasHidden") as HTMLCanvasElement;

        (document.querySelector(".chooseFileFrame .ocrAns") as HTMLImageElement).innerHTML = "";
        (document.querySelector(".ppPlateFrame .ocrAns") as HTMLImageElement).innerHTML = "";

        canvas.width = detectImg.width;
        canvas.height = detectImg.height;
        canvas2.width = detectImg.width;
        canvas2.height = detectImg.height;

        const context = canvas.getContext("2d");
        const context2 = canvas2.getContext("2d");

        context.drawImage(detectImg,0,0,detectImg.width,detectImg.height);
        context2.drawImage(detectImg,0,0,detectImg.width,detectImg.height);

        //let rate = canvas.width / detectImg.width;
        
        /*const input2 = await prepare_input(canvas);
        console.log(input2)*/

        /*const model = await InferenceSession.create("./source/license_plate_detector.onnx");
        const session = await l
        let input = new Tensor(Float32Array.from(input2),[1, 3, 640, 640]);
        const outputs = await model.run({images:input});*/
        //return outputs["output0"].data; 




        /*const session = await new InferenceSession();
        const uri = "./source/license_plate_detector.onnx";
        await session.loadModel(uri);
        
        let input = new Tensor(new Float32Array( input2 ), 'float32', [1, 640, 640, 3]);
        console.log(input)

        const outputMap = await session.run([input]);
        const outputTensor = await outputMap.values().next().value;
        console.log(outputTensor)
        const predictions = await outputTensor.data;*/



        //*const maxPrediction = await Math.max(...predictions);
        //console.log(predictions.indexOf(maxPrediction));
        //let cnn = await characters.charAt(predictions.indexOf(maxPrediction)).toString();*/
        
        //var output = outputs["output0"].data;





        /*const canvas3 = document.querySelector("#origin") as HTMLCanvasElement;
        boxes = process_output(predictions, canvas3.width, canvas3.height);
        draw_boxes(canvas3, boxes, rate);*/



        
        const input = prepare_input(canvas);
        if (!busy) {
            busy = true;
            worker.postMessage(input);
        }
    }

    worker.onmessage = (event:any) => {
        const output = event.data;
        const canvas = document.querySelector("#origin") as HTMLCanvasElement;
        boxes =  process_output(output, canvas.width, canvas.height);
        draw_boxes(canvas, boxes);
        busy = false;
    };


    /*const onmessage = async(event:any) => {
        const input = event.data;
        const output = await run_model(input);
        postMessage(output);
    }
    
    const run_model = async(input:any)=> {
        const model = await InferenceSession.create("./source/license_plate_detector.onnx");
        input = new Tensor(Float32Array.from(input),[1, 3, 640, 640]);
        const outputs = await model.run({images:input});
        return outputs["output0"].data;
    }*/


    const playBtn = document.getElementById("detectBtn");

    function prepare_input(img : any) {
        const canvas = document.createElement("canvas");
        canvas.width = 640;
        canvas.height = 640;
        const context = canvas.getContext("2d");
        context.drawImage(img, 0, 0, 640, 640);
        const data = context.getImageData(0,0,640,640).data;
        const red = [], green = [], blue = [];
        for (let index=0;index<data.length;index+=4) {
            red.push(data[index]/255);
            green.push(data[index+1]/255);
            blue.push(data[index+2]/255);
        }
        return [...red, ...green, ...blue];
    }

    

    function process_output(output:any, img_width:any, img_height:any) {
        let boxes:any = [];
        for (let index=0;index<8400;index++) {
            const [class_id,prob] = [...Array(yolo_classes.length).keys()]
                .map(col => [col, output[8400*(col+4)+index]])
                .reduce((accum, item) => item[1]>accum[1] ? item : accum,[0,0]);
            if (prob < 0.5) {
                continue;
            }
            const label = yolo_classes[class_id];
            const xc = output[index];
            const yc = output[8400+index];
            const w = output[2*8400+index];
            const h = output[3*8400+index];
            const x1 = (xc-w/2)/640*img_width;
            const y1 = (yc-h/2)/640*img_height;
            const x2 = (xc+w/2)/640*img_width;
            const y2 = (yc+h/2)/640*img_height;
            boxes.push([x1,y1,x2,y2,label,prob]);
        }
        boxes = boxes.sort((box1:any,box2:any) => box2[5]-box1[5])
        const result = [];
        while (boxes.length>0) {
            result.push(boxes[0]);
            boxes = boxes.filter((box:any) => iou(boxes[0],box)<0.7 || boxes[0][4] !== box[4]);
        }
        return result;
    }

    function iou(box1:any,box2:any) {
        return intersection(box1,box2)/union(box1,box2);
    }

    function union(box1:any,box2:any) {
        const [box1_x1,box1_y1,box1_x2,box1_y2] = box1;
        const [box2_x1,box2_y1,box2_x2,box2_y2] = box2;
        const box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
        const box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
        return box1_area + box2_area - intersection(box1,box2)
    }

    function intersection(box1:any,box2:any) {
        const [box1_x1,box1_y1,box1_x2,box1_y2] = box1;
        const [box2_x1,box2_y1,box2_x2,box2_y2] = box2;
        const x1 = Math.max(box1_x1,box2_x1);
        const y1 = Math.max(box1_y1,box2_y1);
        const x2 = Math.min(box1_x2,box2_x2);
        const y2 = Math.min(box1_y2,box2_y2);
        return (x2-x1)*(y2-y1)
    }

    function draw_boxes(canvas:any,boxes:any) {
        const ctx = canvas.getContext("2d");
        ctx.strokeStyle = "#ff0000";
        ctx.lineWidth = 3;
        ctx.font = "18px serif";
        boxes.forEach(([x1,y1,x2,y2,label]:any) => {
            ctx.strokeRect(x1,y1,x2-x1,y2-y1);
            ctx.fillStyle = "#ff0000";
            const width = ctx.measureText(label).width;
            ctx.fillRect(x1,y1-20,width+10,25);
            ctx.fillStyle = "#ffffff";
            ctx.fillText(label, x1, y1);
            Crop(canvas,x1,y1,x2-x1,y2-y1);
        });

        
    }

    
    //import { cv } from './module.js';
    
    const { createWorker } = Tesseract;
    let cropRectsArr:any = [];

    const Crop = (can:any,a:any,b:any,c:any,d:any) =>{
        const canvas = document.querySelector("#plate") as HTMLCanvasElement;
        const detectImg = document.querySelector("#originCanvasHidden") as HTMLCanvasElement;
        canvas.width = 1000*(c/d);
        canvas.height = 1000;
        const context = canvas.getContext("2d");
        context.drawImage(detectImg,a+10,b+15,c-20,d-30,0,0,1000*(c/d),1000);



        const canvas2 = document.querySelector("#opencv") as HTMLCanvasElement;
        const canvas3 = document.querySelector("#opencvCanvasHidden") as HTMLCanvasElement;
        /*canvas2.width = 3*c;
        canvas2.height = 3*d;*/






        //const canvas6 = document.querySelector("#plate") as HTMLCanvasElement;
        











        const src = cv.imread("plate");
        let dst = cv.Mat.zeros(25*d, 25*c, cv.CV_8UC3);

        const gray = new cv.Mat();
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

        let M = cv.Mat.ones(5, 5, cv.CV_8U);
        let anchor = new cv.Point(-1, -1);
        // You can try more different parameters
        cv.erode(gray, gray, M, anchor, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());
        cv.dilate(gray, gray, M, anchor, 1, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue());

        const threshold = new cv.Mat();
        cv.threshold(gray, threshold, darkSetting, 255, cv.THRESH_BINARY);

        

        const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(1, 1));
        cv.morphologyEx(threshold, threshold, cv.MORPH_CLOSE, kernel);

        const edges = new cv.Mat();
        cv.Canny(threshold, edges, 50, 100, 3, false);

        //cv.imshow("opencv", threshold);
        //cv.imshow("opencvCanvasHidden", threshold);

        const contours = new cv.MatVector();
        const hierarchy = new cv.Mat();
        cv.findContours(edges, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);

        
        cropRectsArr = [];

        for (let i = 0; i < contours.size(); i++) {
            let cnt = contours.get(i);
            // You can try more different parameters
            let rect = cv.boundingRect(cnt);
            //console.log(rect.height)
            let contoursColor = new cv.Scalar(255, 0, 0);
            let rectangleColor = new cv.Scalar(255, 0, 0);
            if(rect.height > 400 && rect.height < 580 && rect.width < 350 && rect.width > 80){
                //cv.drawContours(threshold, contours, 0, contoursColor, 1, 8, hierarchy, 100);
                let point1 = new cv.Point(rect.x, rect.y);
                let point2 = new cv.Point(rect.x + rect.width, rect.y + rect.height);
                cv.rectangle(edges, point1, point2, rectangleColor, 2, cv.LINE_AA, 0);
                //cv.imshow("opencvCanvasHidden", threshold);
                //console.log(point1.x);
                var obj = {
                    point : point1,
                    width : rect.width,
                    height : rect.height
                }
                cropRectsArr.push(obj)
                
                

            }
            
        }
        
        cv.bitwise_not(threshold, threshold);
        cv.imshow("opencv", edges);
        cv.imshow("opencvCanvasHidden", threshold);

        var link = document.createElement('a');
        link.download = 'filename.png';
        link.href = (document.querySelector("#opencv") as HTMLCanvasElement).toDataURL()
        //link.click();


        GoThrough()















        /*const recognizer = createWorker();

        recognizer.recognize(dst)
        .then((result:any) => {
            console.log(result.text);
        });*/

        const canvas4 = document.querySelector("#opencvCanvasHidden") as HTMLCanvasElement;

        //(document.querySelector("#detectImg") as HTMLImageElement).src = canvas.toDataURL();

        runOCR(canvas4.toDataURL())
            .then((result) => {
                console.log('OCR Result:', result);
                (document.querySelector(".chooseFileFrame .ocrAns") as HTMLImageElement).innerHTML = result
            })
            .catch((error) => {
                console.error('OCR Error:', error);
            });

        

        
        dst.delete(); gray.delete(); threshold.delete(); /*kernel.delete();*/
    }


    const characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    

    const Predict = async(img:any)=>{

        //const session = await InferenceSession.create("./source/cnn_model.onnx");
        const session = await new InferenceSession();
        const uri = "/source/cnn_model.onnx";
        await session.loadModel(uri);
        
        let inputs = new Tensor(new Float32Array( img ), 'float32', [1, 28, 28, 3]);
        console.log(inputs)

        //const feeds = {conv2d_input : new Tensor('float32', new Float32Array( img ),  [1, 28, 28, 3])};

        /*console.log(typeof(img))
        Array.prototype.slice.call(img)
        let er:any = []
        for(var i = 0; i < img.length ; i++){
            if(img[i] >= 200){
                er.push(false)
            }else{
                er.push(true)
            }
        }
        console.log(er)
        
        const feeds = {input : new Tensor('float32',  new Float32Array( er ) ,  [1, 784])};*/

        

        /*const outputMap = await session.run(feeds);
        const outputTensor = outputMap.dense_1.data;
        const predictions = Array.prototype.slice.call(outputTensor);
        //console.log(outputMap)
        const maxPrediction = await Math.max(...predictions);
        let cnn = await characters.charAt(predictions.indexOf(maxPrediction)).toString();*/


        //const outputTensor = await outputMap.values().next().value;




        const outputMap = await session.run([inputs]);
        const outputTensor = await outputMap.values().next().value;
        console.log(outputTensor)
        const predictions = await outputTensor.data;
        const maxPrediction = await Math.max(...predictions);
        console.log(predictions.indexOf(maxPrediction));
        let cnn = await characters.charAt(predictions.indexOf(maxPrediction)).toString();


        /*const outputMap2 = await session2.run([inputs]);
        const outputTensor2 = await outputMap2.values().next().value;
        console.log(outputTensor2)
        const predictions2 = await outputTensor2.data;
        const maxPrediction2 = await Math.max(...predictions2);
        console.log(predictions2.indexOf(maxPrediction2));*/

        let ans = {
            cnn : cnn,
            svc : "2"
        }



        return ans;
    }



    const Predict2 = async(img:any)=>{

        /*const session2 = new InferenceSession();
        const uri2 = "./svc_model_9.onnx";
        await session2.loadModel(uri2);

        let inputs = new Tensor(new Float32Array( img ), 'float32', [1, 28, 28, 3]);
        console.log(inputs)

        
        const outputMap2 = await session2.run([inputs]);
        const outputTensor2 = await outputMap2.values().next().value;
        console.log(outputTensor2)
        const predictions2 = await outputTensor2.data;
        const maxPrediction2 = await Math.max(...predictions2);
        console.log(predictions2.indexOf(maxPrediction2));

        let ans = {
            cnn : "2",
            svc : "3"
        }



        return ans;*/
    }



    

    const GoThrough = async() => {
        
        cropRectsArr.sort((a:any,b:any)=>(
            a.point.x-b.point.x
        ))

        for (let index = 0; index < cropRectsArr.length; index ++) {
            let src2 = await cv.imread('opencvCanvasHidden');
            let cropRect = new cv.Rect(cropRectsArr[index].point.x, cropRectsArr[index].point.y, cropRectsArr[index].width, cropRectsArr[index].height);
            let crop = new cv.Mat();
            crop = src2.roi(cropRect);
            cv.resize(crop, crop, new cv.Size(28, 28));
            //cv.threshold(crop, crop, 120, 0, cv.THRESH_BINARY);
            
            cv.imshow("wordCanvas", crop);
            //cv.imshow("opencv", crop);

            let mat = new cv.Mat(28, 28, 3);
            mat = crop;
            //console.log(mat)
            let row = 28, col = 28;
            //let arr = []
            /*if (mat.isContinuous()) {
                let R = mat.data[row * mat.cols * mat.channels() + col * mat.channels()];
                let G = mat.data[row * mat.cols * mat.channels() + col * mat.channels() + 1];
                let B = mat.data[row * mat.cols * mat.channels() + col * mat.channels() + 2];
                let A = mat.data[row * mat.cols * mat.channels() + col * mat.channels() + 3];
                
            }*/
            let R:any = []
            let G:any = []
            let B:any = []
            let S:any = []
            for(var i = 0 ; i < 28 ; i++){
                //var arr=[];
                for(var j = 0 ; j < 28 ; j++){
                    //var rgb = [];
                    S.push(mat.ucharAt(i, j * mat.channels()));
                    S.push(mat.ucharAt(i, j * mat.channels()+1));
                    S.push(mat.ucharAt(i, j * mat.channels()+2));
                    //S.push(rgb)
                }
                //S.push(arr);
            }
            /*for(var i = 0 ; i < 28 ; i++){
                for(var j = 0 ; j < 28 ; j++){
                    R.push(mat.ucharAt(i, j * mat.channels()));
                    S.push(mat.ucharAt(i, j * mat.channels()));
                }
            }
            for(var i = 0 ; i < 28 ; i++){
                for(var j = 0 ; j < 28 ; j++){
                    B.push(mat.ucharAt(i, j * mat.channels()+1));
                    S.push(mat.ucharAt(i, j * mat.channels()+1));
                }
            }
            for(var i = 0 ; i < 28 ; i++){
                for(var j = 0 ; j < 28 ; j++){
                    G.push(mat.ucharAt(i, j * mat.channels()+2));
                    S.push(mat.ucharAt(i, j * mat.channels()+2));
                }
            }*/
            
            //console.log(S)
            
            let canvas = document.querySelector("#wordCanvas") as HTMLCanvasElement;
            let ctx = canvas.getContext("2d");

            //console.log(ctx.canvas.width,ctx.canvas.height);
            //let ans2 = await Predict2(S);
            //if(index%2 ==0)(document.querySelector(".detectPlateFrame .ocrAns") as HTMLImageElement).innerHTML += ans2.svc;
            //console.log(crop);
            //let ans = await Predict(S);
            let ans = await Predict(S);
            if(index%2 ==0)(document.querySelector(".ppPlateFrame .ocrAns") as HTMLImageElement).innerHTML += ans.cnn;

        };
    }



    const runOCR = async(img:any) => {
        const worker = await createWorker();

        await worker.load();
        await worker.loadLanguage('eng');
        await worker.initialize('eng');
        await worker.setParameters({
            tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
          });
        const { data: { text } } = await worker.recognize(img);
        await worker.terminate();

        return text;
    }



    const ShowImage = (e:React.ChangeEvent<HTMLInputElement>) =>{
        var t = e.target as HTMLInputElement;
        var file = t.files[0];
        //console.log(file);
        var reader = new FileReader();
        reader.onload = function(e) {
            (document.querySelector("#detectImg") as HTMLImageElement).src = e.target.result as string;

        };

        reader.readAsDataURL(t.files[0]);
        const canvas = document.querySelector("#origin") as HTMLCanvasElement;
        const context = canvas.getContext('2d');
        context.clearRect(0, 0, canvas.width, canvas.height);

        boxes = [];

        (document.querySelector(".chooseFileFrame .ocrAns") as HTMLImageElement).innerHTML = "";
        (document.querySelector(".ppPlateFrame .ocrAns") as HTMLImageElement).innerHTML = "";
        
    }

    const yolo_classes = [
        'license plate'
    ];



    /*const preProcess = (ctx: CanvasRenderingContext2D): Tensor =>{
        const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
        const { data, width, height } = imageData;
        const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
        const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [1, 3, width, height]);
        ops.assign(dataProcessedTensor.pick(0, 0, null, null), dataTensor.pick(null, null, 2));
        ops.assign(dataProcessedTensor.pick(0, 1, null, null), dataTensor.pick(null, null, 1));
        ops.assign(dataProcessedTensor.pick(0, 2, null, null), dataTensor.pick(null, null, 0));
        ops.divseq(dataProcessedTensor, 255);
        ops.subseq(dataProcessedTensor.pick(0, 0, null, null), 0.485);
        ops.subseq(dataProcessedTensor.pick(0, 1, null, null), 0.456);
        ops.subseq(dataProcessedTensor.pick(0, 2, null, null), 0.406);
        ops.divseq(dataProcessedTensor.pick(0, 0, null, null), 0.229);
        ops.divseq(dataProcessedTensor.pick(0, 1, null, null), 0.224);
        ops.divseq(dataProcessedTensor.pick(0, 2, null, null), 0.225);
        const tensor = new Tensor(new Float32Array( width * height * 3), 'float32', [1, width, height, 3]);
        (tensor.data as Float32Array).set(dataProcessedTensor.data);
        return tensor;
    }*/

    return (
        <div>

            <div className="title">License Plate Recognition</div>

            <div className="chooseFileFrame">
                <input className="chooseFileBtn" type="file" onChange={(e)=>ShowImage(e)}/>
                <img className="detectImg" id="detectImg" src="" alt="請選擇圖片"/>

                <div className="ocrFrame">
                    <div className="ocrTitle">Tesseract:</div>
                    <div className="ocrAns"></div>
                </div>
            </div>

            <canvas className="origin" id="plate"></canvas>
            <canvas className="origin" id="opencvCanvasHidden"></canvas>
            <canvas className="origin" id="originCanvasHidden"></canvas>
            <canvas className="wordCanvas" id="wordCanvas"></canvas>

            <div className="detectPlateFrame">
                <button className="detectBtn" id="detectBtn" onClick={Detect}>Detect</button>
                <button onClick={Minus}>-</button>
                <span id="darkSetting">亮度調整 : {darkSetting}</span>
                <button onClick={Plus}>+</button>
                
                
                
                <canvas className="plateCanvas" id="origin"></canvas>

                <div className="ocrFrame" style={{"display":"none"}}>
                    <div className="ocrTitle">SVM:</div>
                    <div className="ocrAns"></div>
                </div>


                
            </div>

            <div className="ppPlateFrame">
                <button className="ppBtn" id="ppBtn">Pre Processing</button>
                <canvas className="opencvCanvas" id="opencv"></canvas>

                <div className="ocrFrame">
                    <div className="ocrTitle">CNN:</div>
                    <div className="ocrAns"></div>
                </div>
            </div>


            
            
        </div>
    )
};





/*




<div>
                <input type="file" onChange={(e)=>ShowImage(e)}/>
                <img id="detectImg" src="" alt="請選擇圖片"/>
                
                <button id="detectBtn" onClick={Detect}>Detect</button>
                <canvas id="origin"></canvas>

                <canvas id="plate"></canvas>

                <button id="ppBtn">Pre-Process</button>
                <canvas id="opencv"></canvas>
            </div>





<div className="chooseFileFrame">
                <input className="chooseFileBtn" type="file" onChange={(e)=>ShowImage(e)}/>
                <img className="detectImg" id="detectImg" src="" alt="請選擇圖片"/>
            </div>

            <canvas className="origin" id="plate"></canvas>

            <div className="detectPlateFrame">
                <button className="detectBtn" id="detectBtn" onClick={Detect}>Detect</button>
                <canvas className="plateCanvas2" id="origin"></canvas>
            </div>

            <div className="ppPlateFrame">
                <button className="ppBtn" id="ppBtn">Pre-Process</button>
                <canvas className="opencvCanvas2" id="opencv"></canvas>
            </div>



*/