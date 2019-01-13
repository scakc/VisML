function init(nClasses, nSamples, seed){
    var means = tf.randomNormal([nClasses*2,1], 0, 4, 'float32', seed) // 2C x 1
    var stds = tf.add(tf.randomUniform([nClasses*2,1], 0 , 0.5,'float32', seed), 1) // 2C x 1

    var k, meanx, meany, stdx, stdy, datax, datay,data1, trace, datas, cmeans, cvals, pdata;
    
    for (var i = 0; i < nClasses; i++) {
        k = 2*i;
        meanx = means.dataSync()[k]
        meany = means.dataSync()[k+1]
        stdx = stds.dataSync()[k]
        stdy = stds.dataSync()[k+1]

        datax = tf.randomNormal([nSamples,1], meanx, stdx); // S x 1
        datay = tf.randomNormal([nSamples,1], meany, stdy); // S x 1


        data1 = tf.concat([datax, datay], axis = 1); // S x 2

        trace = {
        x: Array.from(data1.slice([0, 0],[nSamples,1]).dataSync()),
        y: Array.from(data1.slice([0, 1],[nSamples,1]).dataSync()),
        mode : 'markers',
        type : 'scatter',
        name : 'class ' + i,
        marker :{
            size :6,
        }
        };


        if(i==0){
            datas = data1;
            cmeans = data1.mean(axis=0).reshape([1,2]); // 1 x 2
            cvals = tf.mul(tf.ones([data1.shape[0], 1]), i);
            pdata = [trace]
        }else{
            datas = tf.concat([datas,data1], axis = 0); // SC x 2 finally
            cmeans = tf.concat([cmeans, data1.mean(axis= 0).reshape([1,2])], axis = 0); // C x 2
            cvals = tf.concat([cvals, tf.mul(tf.ones([data1.shape[0], 1]), i)], axis = 0); // CS x 1
            pdata = pdata.concat(trace);
        }

    }




    layout = { 
      title: "Visualization",
      font: {size: 18}
    };


    

    Plotly.newPlot('myDiv', pdata, layout, {responsive: true});

    return [datas, cmeans, cvals, pdata];

}



function knnclassify(){

    var N, k , mx, my, mean, dist, sdist, labels, trace, layout, kval;

    var kval = parseInt(document.getElementById("kval").value);

    var N = grid.shape[0];
    var CS = datas.shape[0];

    var X_data_train = tf.concat([datas, tf.ones([CS,1])], axis = 1);
    var X_data_test = tf.concat([grid, tf.ones([N,1])], axis = 1);


    
    var w = tf.randomNormal([1,3], 0, 4,'float32');

    var labels = tf.greater(tf.dot(w, tf.transpose(X_data_test)), 0);

    var count = 0;


    var y_hat, y_hat_norm, grad_w,trace,layout,cData;

    while(count < 500){

        y_hat = sigmoid(tf.dot(w, tf.transpose(X_data_train)));
        y_hat_norm = tf.sub(tf.transpose(y_hat), cvals);

        grad_w = tf.div(tf.neg(tf.sum(tf.mul(y_hat_norm, X_data_train), axis = 0)), CS);

        w = tf.sub(w,tf.transpose(tf.mul(0.01, grad_w)));
        

        count = count + 1;

        labels = tf.greaterEqual(tf.dot(w, tf.transpose(X_data_test)), 0);

        // setTimeout(function () {
        //     console.log(w.dataSync());
        //     // updateplot('myDiv', {z : labels.dataSync()});
        // }, 3000);

        // requestAnimationFrame(updateplot);


    }


    labels = tf.greaterEqual(tf.dot(w, tf.transpose(X_data_test)), 0);

    trace = {
        x: Array.from(grid.slice([0, 0],[N,1]).dataSync()),
        y: Array.from(grid.slice([0, 1],[N,1]).dataSync()),
        z: Array.from(labels.dataSync()),
        mode : 'markers',
        opacity : 0.3 ,
        type : 'heatmap',
        name : '',
        showscale: false,
        colorscale : [[0, 'rgb(166,206,227)'], [0.25, 'rgb(31,120,180)'], 
        [0.45, 'rgb(178,223,138)'], [0.65, 'rgb(51,160,44)'], [0.85, 'rgb(251,154,153)'], [1, 'rgb(227,26,28)']]
    };


    layout = { 
      title: "Visualizing Classification",
      font: {size: 18}
    };

    cData = pdata.concat(trace);
    Plotly.react('myDiv', cData, layout, {responsive: true});


}


document.getElementById("init").onclick = function() {varys = cINIT()
    nSamples = varys[1];
    nClasses = varys[0];
    seed = varys[2];
};
document.getElementById("cfy").onclick = function() {knnclassify()};


function cINIT(nClasses, nSamples, seed) {
    if(nClasses){
        ;
    }else{
        nClasses = 2;//document.getElementById("nClasses").value;
        nSamples = document.getElementById("nSamples").value;
        seed = document.getElementById("seed").value;

        nSamples = parseInt(nSamples);
        nClasses = parseInt(nClasses);
        seed = parseInt(seed);

    }
    cache = init(nClasses, nSamples, seed);

    datas = cache[0];
    cmeans = cache[1];
    cvals = cache[2];

    pdata = cache[3];

    // alert(nSamples);

    xymin = tf.min(datas, axis = 0).dataSync();
    xymax = tf.max(datas, axis = 0).dataSync();

    xmin = xymin[0];
    ymin = xymin[1];
    xmax = xymax[0];
    ymax = xymax[1];

    grid = grid2dflat(xmin, xmax, ymin, ymax, 100, 100);


    return [nClasses, nSamples, seed];

}

function cADD() {
    
}
var cmap = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
var nClasses = 2;
var nSamples = 50;
var seed = 5;
var datas,cmeans,cvals,xmin,xmax,ymin,ymax, grid;

cINIT(nClasses, nSamples,seed);

