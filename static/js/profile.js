function checkStartEnd(input1, input2) {
    input2.min = input1.value || '0:00';
}

function addInterval(input) {
    selector = $('#' + input);
    // The initial size is 1. This variable refers to the new size of the list, in terms of intervals (up to 5).
    size = selector.children().length + 1;
    if (size <= 5) {
        selector.append(`<div><input type="time" oninput="checkStartEnd(this, this.nextSibling.nextSibling)"
            name="${input}Start${size}" id="${input}Start${size}"/> to <input type="time"
            oninput="checkStartEnd(this, this.nextSibling.nextSibling)" name="${input}End${size}"
            id="${input}End${size}"/><br></div>`);
    }
}

function removeInterval(input) {
    children = $('#' + input).children();
    size = children.length;
    if (size > 1) {
        children.last().remove();
    }
}