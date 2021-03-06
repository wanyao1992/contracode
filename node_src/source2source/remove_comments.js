var Terser = require("terser");

var options = {
    compress: false,
    mangle: false,
    output: {
        beautify: true,
        comments: false,
    }
};

module.exports = (js_src, {prob = 0.5}) => {
    if (Math.random() < prob) {
        return Terser.minify(js_src, options).code;
    }
    else {
        return js_src;
    }
}
