//第一个图表对应的SVG
var svg = d3.select("#barchart")
            .append("svg")
                .attr("width", 590)
                .attr("height", 400)
                .attr("id", "bar");
d3.csv("/baseball_4.csv", function(data) {

    //绘制单个棒球手的图表
    function plotname(name) {
        //绘制一个指标的图表，包括颜色编码
        function plotbar(x, y, width, height, measure, name, tmin, tmax, legend) {
            data_f = dimple.filterData(data, "Name", [name]);
            var chart = new dimple.chart(svg, data_f);
            chart.setBounds(x, y, width, height);
            var x = chart.addMeasureAxis("x", measure);
            x.overrideMax = tmax;
            x.overrideMin = tmin;
            var y = chart.addCategoryAxis("y", "name");
            chart.addSeries("Hand", dimple.plot.bar);
            y.hidden = true;
            chart.assignColor("Right", "#629FEC");
            chart.assignColor("Both", "#F4AA4A");
            chart.assignColor("Left", "#F66C77");
            if (legend == true) {
                chart.addLegend(180, 10, 360, 20, "right");
            }
            chart.draw();

        }
        //因为只要第一行图表的颜色图例，所以只有第一个图表的legend参数为true
        plotbar(100, 30, 430, 30, "Weight(Pound)", name, 0, 200, true);
        plotbar(100, 120, 430, 30, "Height(Inch)", name, 0, 80, false);
        plotbar(100, 210, 430, 30, "Avg(%)", name, 0, 0.5, false);
        plotbar(100, 300, 430, 30, "HR", name, 0, 600, false);

    };

    //将数据中的名字全部加载到下拉框的备选项中
    var select = d3.select("select");

    select.selectAll("option")
        .data(data)
        .enter()
        .append("option")
        .attr("value", function(d) {
            return d.Name;
        })
        .text(function(d) {
            return d.Name;
        })
    //初始绘图
    plotname("Tom Brown");

    //根据下拉框的改变绘制不同的图表。
    d3.select("select")
        .on("change", function() {

            var selectv = document.getElementById('nlist');
            var index = selectv.selectedIndex;
            var val = selectv.options[index].text;
            svg.selectAll("*").remove();            
            plotname(val);
        })
});

//第二个图表对应的SVG
var svg2 = d3.select("#piechart")
            .append("svg")
                .attr("width", 1300)
                .attr("height", 400)
                .attr("id", "pie");


d3.csv("baseball_group.csv", function(data){
    //绘制饼图
    var chart = new dimple.chart(svg2, data);
    chart.setBounds(20, 20, 460, 360);
    chart.addMeasureAxis("p", "Count");
    chart.addSeries("Hand", dimple.plot.pie);
    chart.addLegend(500, 20, 90, 300, "left");
    chart.draw();
    //分用手习惯绘制单个指标的图表
    function plotcat(x, y, width, height, measure, alia, tmin, tmax)
    {  
        var chart2 = new dimple.chart(svg2, data);
        chart2.setBounds(x, y, width, height);
        var y = chart2.addMeasureAxis("y", measure);
        var x = chart2.addCategoryAxis("x", "Hand");
        y.overrideMax = tmax;
        y.overrideMin = tmin;
        y.title = alia
        x.hidden = true;
        chart2.addSeries("Hand", dimple.plot.bar);
        chart2.draw();
    }
    //绘图
    plotcat(600, 20, 120, 360, "Height(Inch)", "Average Height", 50, 73);
    plotcat(780, 20, 120, 360, "Weight(Pound)", "Average Weight", 100, 200);
    plotcat(960, 20, 120, 360, "Avg(%)", "Average Avg", 0, 0.3);
    plotcat(1140, 20, 120, 360, "HR", "Average HR", 0, 100);

})

