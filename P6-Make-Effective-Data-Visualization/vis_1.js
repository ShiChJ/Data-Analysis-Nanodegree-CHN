var svg = new dimple.newSvg("#chartContainer", 590, 400);
d3.csv("/baseball_3.csv", function(data) {

    function plotname(name) {

        function plotbar(x, y, height, width, measure, name, tmin, tmax) {
            data_f = dimple.filterData(data, "name", [name]);
            var chart = new dimple.chart(svg, data_f);
            chart.setBounds(x, y, height, width);
            var x = chart.addMeasureAxis("x", measure);
            x.overrideMax = tmax;
            x.overrideMin = tmin;
            var y = chart.addCategoryAxis("y", "name");
            chart.addSeries("Hand", dimple.plot.bar);
            y.hidden = true;
            chart.assignColor("Right", "red");
            chart.assignColor("Both", "blue");
            chart.assignColor("Left", "green");
            chart.draw();

        }

        plotbar(100, 30, 430, 30, "weight", name, 0, 200);
        plotbar(100, 120, 430, 30, "height", name, 0, 80);
        plotbar(100, 210, 430, 30, "avg", name, 0, 0.5);
        plotbar(100, 300, 430, 30, "HR", name, 0, 600);

    };
    var select = d3.select("select");

    select.selectAll("option")
        .data(data)
        .enter()
        .append("option")
        .attr("value", function(d) {
            return d.name;
        })
        .text(function(d) {
            return d.name;
        })

   plotname("Tom Brown")


    d3.select("select")
        .on("change", function() {

            var selectv = document.getElementById('nlist');
            var index = selectv.selectedIndex;
            var val = selectv.options[index].text;
            svg.selectAll("*").remove();            
            plotname(val);
        })




});