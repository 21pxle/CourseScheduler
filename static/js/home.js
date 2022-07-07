/* This function "ties" the minimum value of the second input
 to the value of the first input to enforce the requirement that the ranges have. */
function checkStartEnd(input1, input2) {
    input2.min = input1.value || 0;
}

function clearFilters() {
    $("input").not('#apply').val('');
}