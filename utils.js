
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

function grid1dflat(xmin,xmax,nX){
    var xsteps = (xmax - xmin)/nX;
    var xvals = tf.range(xmin,xmax,xsteps).reshape([-1,1]);
    return xvals;
    //
}