function genDates(date, horizon, unit){

    var dates = []
    var date = Date.parse(date)

    if (unit == "w"){

        var week = 6.04e+8 //Week in milliseconds
        var day = 8.64e+7 //Day in milliseconds
        
        for(var i=0; i<horizon; i++){

            var tmp = new Date(date + (week*i)+ day)
            var tmp_str = tmp.getFullYear() + "-"  +(tmp.getMonth() + 1) + "-" + tmp.getDate()

            dates.push(tmp_str)
        }

    }else{

        var getDaysInMonth = function(month,year) {
            //Day 0 is the last day in the previous month
           return new Date(year, month, 0).getDate();
          };
          
        date = new Date(date)
        var tmp_str = date.getFullYear() + "-"  +(date.getMonth() + 1) + "-" +  getDaysInMonth((date.getMonth() + 1), date.getFullYear())
        dates.push(tmp_str)

        for(var i=0;i<horizon-1; i++){
            var tmp = new Date(date + date.setMonth(date.getMonth()+1))
            var tmp_str = tmp.getFullYear() + "-"  +(tmp.getMonth() + 1) + "-" + getDaysInMonth((tmp.getMonth() + 1), tmp.getFullYear())

            dates.push(tmp_str)
        }

    }

    return dates

}

function doThePlots(){
    
    var elements = document.querySelectorAll("#dataTable .selected")

    traces = []

    start = document.getElementById("startInput").value;
    end = document.getElementById("endInput").value;
    horizon = parseInt(document.getElementById("horizonInput").value);
    unit = document.getElementById("unitInput").value;
    forecastStartInput = document.getElementById("forecastStartInput").value;
    predictHorizon = document.getElementById("predictHorizon").value;

    for(var i=0; i<elements.length;i++){
        
        var row = elements[i].children
        var prodName = row[0].innerHTML

        tmp = []

        for(var j=1; j<row.length; j++){

            var val = parseFloat(row[j].innerHTML);
            tmp.push(val)

        }

        var trace = {
            type: "scatter",
            name: prodName,
            mode: "lines",
            x: genDates(start, horizon, unit),
            y: tmp
        }

        traces.push(trace)

    }
    
    tmp = new Date(Date.parse(forecastStartInput))
    a = tmp.getFullYear() + "-"  +(tmp.getMonth() + 1) + "-" + tmp.getDate()

    b = genDates(start, horizon, unit).slice(-1)[0]

    var shapes;
    if(predictHorizon != 0){
        shapes = [{
            type: 'rect',
            // x-reference is assigned to the x-values
            xref: 'x',
            // y-reference is assigned to the plot paper [0,1]
            yref: 'paper',
            x0: a,
            y0: 0,
            x1: b,
            y1: 1,
            fillcolor: '#d3d3d3',
            opacity: 0.2,
            line: {
                width: 0
            }
        }]
    }else{
        shapes = []
    }

    var layout ={

        title: {
            text:"Products' plots comparison",
            font: {
            family: 'Courier New, monospace',
            size: 24
            },
            xref: 'paper',
            x: 0.05,
        },
        autosize:false, 
        width:1100, 
        height: 600,
        shapes: shapes

    }

    Plotly.newPlot('tmpPlotDiv', traces, layout);



    
}


document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function(){
        table.order( [ 1, 'asc' ] ).draw();
       }, 1000);

    
 }, false);


 document.getElementById('sidebar-toggle').addEventListener('click', function(){
    setTimeout(function(){
        table.order( [ 1, 'asc' ] ).draw();
       }, 500);

 })