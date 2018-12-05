function init(nClasses, nSamples, seed){
    var means = tf.randomNormal([nClasses*2,1], 0, 5,'float32', seed) // 2C x 1
    var stds = tf.add(tf.randomNormal([nClasses*2,1], 0 , 0.5,'float32', seed), 1) // 2C x 1

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
            cvals = tf.tensor1d([i]);
            pdata = [trace]
        }else{
            datas = tf.concat([datas,data1], axis = 0); // SC x 2 finally
            cmeans = tf.concat([cmeans, data1.mean(axis= 0).reshape([1,2])], axis = 0); // C x 2
            cvals = tf.concat([cvals, tf.tensor1d([i])]); // C x 1
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

function protoclassify(){

    var N, k , mx, my, mean, dist, sdist, labels, trace, layout;

    N = grid.shape[0];

    meandata = cmeans.dataSync();
    for (var i = 0; i < cvals.shape[0]; i++) {
        k = 2*i;
        mx = meandata[k];
        my = meandata[k+1];
        mean = tf.tensor1d([mx,my]).reshape([1,-1]);
        dist = tf.sum(tf.square(tf.sub(grid,mean)), axis = 1).reshape([N,1]);
        if(i==0){
            sdist = dist;
        }else{
            sdist = tf.concat([sdist, dist], axis = 1);
        }

    }

    labelsindx = tf.argMin(sdist, axis = 1);
    labels = tf.gather(cvals, labelsindx);

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

    meanplot = {
    x: Array.from(cmeans.slice([0, 0],[nClasses,1]).dataSync()),
    y: Array.from(cmeans.slice([0, 1],[nClasses,1]).dataSync()),
    mode : 'markers',
    type : 'scatter',
    marker : {
        size : 10,
        color : 'rgba(50, 250, 150, .8)',
        line : {width : 2},
    },
    name : 'Means',
    };

    layout = { 
      title: "Visualizing Classification",
      font: {size: 18}
    };

    cData = pdata.concat(trace).concat(meanplot);

    Plotly.newPlot('myDiv', cData, layout, {responsive: true});


}

function grid2dflat(xmin,xmax,ymin,ymax, nX,nY){
    var xsteps = (xmax - xmin)/nX;
    var ysteps = (ymax - ymin)/nY;
    var xvals = tf.range(xmin,xmax,xsteps).reshape([-1,1]);
    var yvals = tf.range(ymin,ymax,ysteps).reshape([-1,1]);
    var xn = xvals.shape[0]
    var yn = yvals.shape[0]
    var xgrid = tf.tile(xvals, [1, yn]).reshape([xn*yn, 1]);
    var ygrid = tf.transpose(tf.tile(yvals, [1, xn])).reshape([xn*yn, 1]);
    var grid = tf.concat([xgrid, ygrid], axis = 1);
    return grid;
}


document.getElementById("init").onclick = function() {varys = cINIT()
    nSamples = varys[1];
    nClasses = varys[0];
    seed = varys[2];
};
document.getElementById("cfy").onclick = function() {protoclassify()};


function cINIT(nClasses, nSamples, seed) {
    if(nClasses){
        ;
    }else{
        nClasses = document.getElementById("nClasses").value;
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
var seed = 590;
var datas,cmeans,cvals,xmin,xmax,ymin,ymax, grid;

cINIT(nClasses, nSamples,seed);

