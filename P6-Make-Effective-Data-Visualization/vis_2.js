var svg = new dimple.newSvg("#chartContainer", 590, 400);
d3.csv("/baseball_4.csv", function(data) {

    //绘制单个棒球手的图表
    function plotname(name) {
        //绘制一个指标的图表，包括颜色编码
        function plotbar(x, y, height, width, measure, name, tmin, tmax, legend) {
            data_f = dimple.filterData(data, "Name", [name]);
            var chart = new dimple.chart(svg, data_f);
            chart.setBounds(x, y, height, width);
            var x = chart.addMeasureAxis("x", measure);
            x.overrideMax = tmax;
            x.overrideMin = tmin;
            var y = chart.addCategoryAxis("y", "name");
            chart.addSeries("Hand", dimple.plot.bar);
            y.hidden = true;
            chart.assignColor("Right", "#F84B4B");
            chart.assignColor("Both", "#4141F5");
            chart.assignColor("Left", "#3CD53C");
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