function init(nClasses, nSamples, seed){
    var means = tf.randomNormal([nClasses*2,1], 0, 0.1,'float32', seed) // 2C x 1
    var stds = tf.add(tf.randomNormal([nClasses*2,1], 0 , 0.1,'float32', seed), 1) // 2C x 1

    var k, meanx, meany, stdx, stdy, datax, datay,data1, trace, datas, cmeans, cvals, pdata;
    
    for (var i = 0; i < nClasses; i++) {
        k = 2*i;
        meanx = means.dataSync()[k];
        meany = means.dataSync()[k+1];
        stdx = 5;
        stdy = 0;

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

function Gfit(){



    CS = datas.shape[0]
    X_tr = datas.slice([0,0], [CS,1]);
    X_bar = X_tr.mean();


    mu = X_bar.dataSync()[0];
    sigma = tf.sqrt(tf.div(tf.sum(tf.square(tf.sub(X_tr,X_bar))), CS)).dataSync()[0];
    grid = grid1dflat(mu-20, mu+20, 100);



    N = grid.shape[0];
    X = grid.slice([0,0], [N,1]);
    p_vals = gaussian(X, mu, sigma);

    trace = {
        x: Array.from(X.dataSync()),
        y: Array.from(p_vals.dataSync()),
        mode : 'line',
        type : 'line',
        name : 'fit',
        marker :{
            size :6,
        }
    };

    layout = { 
      title: "Gaussian Fit MLE",
      font: {size: 18}
    };


    Plotly.newPlot('fit', [trace], layout, {responsive: true});



    mu0 = parseInt(document.getElementById("mu0").value);    
    sigma0 = parseInt(document.getElementById("sig0").value);
    sigma = parseInt(document.getElementById("sig").value);


    // mu0 = 0;
    // sigma0 = 1;


    sigma_n = tf.sqrt(tf.div(1,tf.add(tf.div(1, tf.square(sigma0)), tf.div(CS, tf.mul(1, tf.square(sigma))))));
    mu_n = sigma_n.square().mul(tf.add(tf.div(mu0, tf.square(sigma0)), (X_bar.mul(CS)).div(tf.square(sigma)))).dataSync()[0];
    grid = grid1dflat(mu0-5*sigma0, mu0+5*sigma0, 100);

    N = grid.shape[0];
    X = grid.slice([0,0], [N,1]);
    g_vals = gaussian(X, mu_n, sigma_n.dataSync()[0]);

    trace_1 = {
        x: Array.from(X.dataSync()),
        y: Array.from(g_vals.dataSync()),
        mode : 'line',
        type : 'line',
        name : 'postr',
        marker :{
            size :6,
        }
    };

    pr_vals = gaussian(X, mu0, sigma0);

    trace_2 = {
        x: Array.from(X.dataSync()),
        y: Array.from(pr_vals.dataSync()),
        mode : 'line',
        type : 'line',
        name : 'prior',
        marker :{
            size :6,
        }
    };

    layout = { 
      title: "Gaussian Posterior",
      font: {size: 18}
    };

    Plotly.newPlot('pos', [trace_1, trace_2], layout, {responsive: true});

    

}



document.getElementById("init").onclick = function() {varys = cINIT()
    nSamples = varys[1];
    nClasses = varys[0];
    seed = varys[2];
};
document.getElementById("cfy").onclick = function() {Gfit()};


function cINIT(nClasses, nSamples, seed) {
    if(nClasses){
        ;
    }else{
        nClasses = 1;//document.getElementById("nClasses").value;
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


    return [nClasses, nSamples, seed];
}

function cADD() {
    
}
var cmap = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
var nClasses = 1;
var nSamples = 50;
var seed = 590;
var datas,cmeans,cvals,xmin,xmax,ymin,ymax, grid;

cINIT(nClasses, nSamples,seed);

