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
        datay = tf.mul(tf.randomNormal([1,1],0,1),tf.add(tf.square(datax), tf.mul(tf.randomNormal([1,1],0,1),tf.cos(tf.mul(tf.randomNormal([1,1],0,5),datax))))); // S x 1


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

    kval = parseInt(document.getElementById("kval").value);

    
    CS = datas.shape[0];

    var r_vals = datas.slice([0,1], [CS,1]).reshape([-1])
    var r_data = datas.slice([0,0], [CS,1]).reshape([-1])
    var r_grid = grid
    
    N = r_grid.shape[0];

    distance = tf.square(tf.sub(r_data, r_grid.expandDims(1)))

    varsg = tf.topk(tf.neg(distance), kval);

    

    for (var i = 0; i < kval; i++) {

        top_k_label = tf.gather(r_vals, tf.slice(varsg.indices, [0,0,i],[N,1,1]).reshape([-1]));
        if(i == 0){
            sumvec = top_k_label;
        }else{
            sumvec = tf.add(sumvec, top_k_label);
        }
    }


    labels = tf.div(sumvec, kval);

    trace = {
    x: Array.from(grid.dataSync()),
    y: Array.from(labels.dataSync())
    };


    layout = { 
      title: "Visualizing Regression",
      font: {size: 18}
    };

    cData = pdata.concat(trace);

    Plotly.newPlot('myDiv', cData, layout, {responsive: true});


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
        nClasses = 1;
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

    grid = grid1dflat(xmin, xmax, 100);

    return [nClasses, nSamples, seed];
}

function cADD() {
    
}

function grid1dflat(xmin,xmax,nX){
    var xsteps = (xmax - xmin)/nX;
    var xvals = tf.range(xmin,xmax,xsteps).reshape([-1,1]);
    return xvals;
    //
}


var cmap = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
var nClasses = 1;
var nSamples = 50;
var seed = 5;
var datas,cmeans,cvals,xmin,xmax,ymin,ymax, grid;

cINIT(nClasses, nSamples,seed);

