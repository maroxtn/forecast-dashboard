import * as $ from 'jquery';
import 'datatables';

export default (function () {
  var table = $('#dataTable').DataTable({
    "scrollX": true
  });

  $('#dataTable tbody').on('click', 'tr', function () {
    $(this).toggleClass('selected');

    var count = table.rows('.selected').data().length;

    if(count > 0)
      $("#dataTable_length button").css("opacity","1")
    else
    $("#dataTable_length button").css("opacity","0")

  } );

  var htmltext = ""
  htmltext += $("#hiddenForm").html()

  $("#dataTable_length").html(htmltext)

  $("#dataTable_length button").on('click',function(){ 

    doThePlots()

    $("#plot").modal('toggle')
    
  })

  window.table = table

}());
