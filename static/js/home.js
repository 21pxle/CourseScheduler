/* This function "ties" the minimum value of the second input
 to the value of the first input to enforce the requirement that the ranges have. */
function checkStartEnd(input1, input2) {
    if (input1.type == "time") {
        input2.min = input1.value || '00:00';
    } else {
        input2.min = input1.value || 0;
    }
}

function applySearchFilter() {
    input = $("#filter").val();
    results1 = $("#course option:contains(" + input + ')');
    results2 = $("#selected option:contains(" + input + ')');

    $(".search option").not(':contains(' + input + ')').hide();
    $(".search option:contains(" + input + ')').show();

    if (results1.length) {
        $("#course").show();
    } else {
        $("#course").hide();
    }
    if (results2.length) {
        $("#selected").show();
    } else {
        $("#selected").hide();
    }
}

function addCourses() {
    input = $("#course option:selected");
    results = $("#course option").not(":selected").not(":hidden");

    if (!results.length) {
        $("#course").hide();
    }
    if (input.length) {
        $("#selected").show();
    }
    $("#selected").append(input);
    $("#course option:selected").remove();
}

function removeCourses() {
    input = $("#selected option:selected");
    results = $("#selected option").not(":selected").not(":hidden");

    if (!results.length) {
        $("#selected").hide();
    }
    if (input.length) {
        $("#course").show();
    }
    $("#course").append(input);
    $("#selected option:selected").remove();
}


function clearFilters() {
    $("input").not('#apply').val('');
}

function clearSearchFilter() {
    filter = $("#filter").val();
    results1 = $("#course option");
    results2 = $("#selected option");

    $(".search option").show();

    if (results1.length) {
        $("#course").show();
    }
    if (results2.length) {
        $("#selected").show();
    }
}