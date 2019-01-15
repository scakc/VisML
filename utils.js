
function sigmoid(z){

    return tf.div(1 , tf.add(1, tf.exp(tf.neg(z))));
    
}

function gaussian(z, mu, sigma){

    var varn = tf.square(sigma);
    var pi = 3.1415926;
    var denom = tf.pow(tf.mul(2*pi, varn), 0.5);
    var expo = tf.mul(tf.div(tf.square(tf.sub(z,mu)),varn), -0.5);
    var c = tf.div(tf.exp(expo), denom);
    return c;

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

function grid1dflat(xmin,xmax,nX){
    var xsteps = (xmax - xmin)/nX;
    var xvals = tf.range(xmin,xmax,xsteps).reshape([-1,1]);
    return xvals;
    //
}


// calculate the determinant of a matrix m
function det(m) {
    return tf.tidy(() => {
        const [r, _] = m.shape
        if (r === 2) {
            const t = m.as1D()
            const a = t.slice([0], [1]).dataSync()[0]
            const b = t.slice([1], [1]).dataSync()[0]
            const c = t.slice([2], [1]).dataSync()[0]
            const d = t.slice([3], [1]).dataSync()[0]
            result = a * d - b * c
            return result

        } else {
            let s = 0;
            rows = [...Array(r).keys()]
            for (let i = 0; i < r; i++) {
                sub_m = m.gather(tf.tensor1d(rows.filter(e => e !== i), 'int32'))
                sli = sub_m.slice([0, 1], [r - 1, r - 1])
                s += Math.pow(-1, i) * det(sli)
            }
            return s
        }
    })
}

// the inverse of the matrix : jordan-gauss method
function invertM(m) {
    return tf.tidy(() => {
        if (det(m) === 0) {
            console.log('Zero det')
            return
        }
        const [r, _] = m.shape
        let inv = m.concat(tf.eye(r), 1)
        rows = [...Array(r).keys()]
        for (let i = 0; i < r; i++) {
            inv = tf.tidy(() => {
                for (let j = i + 1; j < r; j++) {
                    const elt = inv.slice([j, i], [1, 1]).as1D().asScalar()
                    const pivot = inv.slice([i, i], [1, 1]).as1D().asScalar()
                    let newrow
                    if (elt.dataSync()[0] !== 0) {
                        newrow = inv.gather(tf.tensor1d([i], 'int32')).sub(inv.gather(tf.tensor1d([j], 'int32')).div(elt).mul(pivot)).as1D()
                        const sli = inv.gather(rows.filter(e => e !== j))
                        const arr = []
                        if (j === 0) {
                            arr.push(newrow)
                        }
                        sli.unstack().forEach((t, ind) => {
                            if (ind !== j) {
                                arr.push(t)
                            } else {
                                arr.push(newrow)
                                arr.push(t)
                            }
                        })
                        if (j === r - 1) {
                            arr.push(newrow)
                        }
                        inv = tf.stack(arr)
                    }
                }
                return inv
            })
        }
        const trian = tf.unstack(inv)
        len = trian.length
        trian[len - 1] = trian[len - 1].div(trian[len - 1].slice(trian[len - 1].shape[0] - 1, 1).asScalar())
        for (let i = r - 2; i > -1; i--) {
            for (j = r - 1; j > i; j--) {
                trian[i] = trian[i].sub(trian[j].mul(trian[i].slice(j, 1).asScalar()))
            }
            trian[i] = trian[i].div(trian[i].slice(i, 1).asScalar())
        }
        return tf.split(tf.stack(trian), 2, 1)[1]
    })
}