var color = 'black';
var rectangles = [];

function init() {
  generateClassSelector();
}

function clearCanvas() {
    rectangles = [];
    redrawCanvas();
}

function undo() {
    var lastRect = rectangles.pop();
    if (lastRect) {
        $('#canvas').children().last().remove();
    }
}

function getSDImages() {
    $.ajax({
        type: 'POST',
        url: '/get_sd_images',
        contentType: 'application/json',
        data: JSON.stringify({rectangles: rectangles}),
        success: function(data) {
            // create a new div element
            var imageDiv = document.createElement('div');
            imageDiv.id = 'image-container';
            imageDiv.style.border = '1px solid black';

            // create a new img element and set its src attribute
            var image = document.createElement('img');
            console.log(data)
            url = '/image/' + data['image_path']
            image.src = url;

            // append the img element to the new div element
            imageDiv.appendChild(image);

            // append the new div element to the body
            document.body.appendChild(imageDiv);
        }
    });
}

function createRectangle(x1, y1, x2, y2, color, class_name=$("input[name='class']:checked").val()) {
    var rect = $('<div>').css({
        position: 'absolute',
        left: Math.min(x1, x2) + 'px',
        top: Math.min(y1, y2) + 'px',
        width: Math.abs(x2 - x1) + 'px',
        height: Math.abs(y2 - y1) + 'px',
        border: '2px solid ' + color
    });

    var colorbox = $('<div>').css({
        position: 'absolute',
        left: '0px',
        top: '0px',
        width: '50px',
        height: '20px',
        background: 'none',
        color: color,
        'text-align': 'center',
        'font-size': '12px',
        'line-height': '20px'
    }).text(class_name);

    rect.append(colorbox);
    var rect_config = {
        x1: Math.max(0, Math.min(x1, x2)),
        y1: Math.max(0, Math.min(y1, y2)),
        x2: Math.min(511, Math.max(x1, x2)),
        y2: Math.min(511, Math.max(y1, y2)),
        color: color,
        class: class_name
    };

    return {rect, rect_config};
}

function drawRectangle(rect) {
    $('#canvas').append(rect);
}

function updateRectangle(rect, x1, y1, x2, y2) {
    rect.css({
        left: Math.min(x1, x2) + 'px',
        top: Math.min(y1, y2) + 'px',
        width: Math.abs(x2 - x1) + 'px',
        height: Math.abs(y2 - y1) + 'px'
    });

    return rect;
}

function redrawCanvas() {
    $('#canvas').empty();
    for (var i = 0; i < rectangles.length; i++) {
        var rect = rectangles[i];
        res = createRectangle(
            rect.x1, rect.y1, rect.x2, rect.y2, rect.color, rect.class
        );
        thisRect = res.rect
        drawRectangle(thisRect)
    }
}

$(function() {
    var isDrawing = false;

    $('#canvas').mousedown(function(event) {
        startX = event.offsetX;
        startY = event.offsetY;
        isDrawing = true;
        res = createRectangle(startX, startY, startX, startY, color);
        currentRect = res.rect;
        currentRectConfig = res.rect_config;
    });

    $('#canvas').mousemove(function(event) {
        if (isDrawing) {
            // clear the canvas and redraw any existing rectangles
            redrawCanvas()
            // get the current end point and draw a new rectangle
            endX = event.offsetX;
            endY = event.offsetY;
            currentRect = updateRectangle(currentRect, startX, startY, endX, endY);
            drawRectangle(currentRect)
        }
    });

    $('#canvas').mouseup(function(event) {
        if (isDrawing) {
            isDrawing = false;

            // get the final end point and create a new rectangle
            // endX = event.offsetX;
            // endY = event.offsetY;
            res = createRectangle(startX, startY, endX, endY, color);
            currentRect = res.rect;
            currentRectConfig = res.rect_config;
            rectangles.push(currentRectConfig);
        }
    });

    $('#clearbtn').click(clearCanvas);
    // Add event listener to Plot button
    $('#submit').click(getSDImages);
    // redraw the canvas on page load to display any existing rectangles
    redrawCanvas();
});

function generateClassSelector() {
  // Make a GET request to read the labels file
  const xhr = new XMLHttpRequest();
  xhr.open('GET', '/static/doc/labels.txt');
  xhr.onreadystatechange = function() {
    if (xhr.readyState === 4 && xhr.status === 200) {
      // Split the labels into an array
      const labels = xhr.responseText.trim().split('\n');
      const classes = {};
      // Loop through each label to generate a unique color and create a radio button
      for (let i = 1; i < labels.length; i++) {
        const [classId, className] = labels[i].split(': ');
        classes[className] = `#${(Math.random()*0xFFFFFF<<0).toString(16).padStart(6, '0')}`;
        const input = document.createElement('input');
        input.type = 'radio';
        input.name = 'class';
        input.value = className;
        const label = document.createElement('label');
        label.htmlFor = className;
        label.innerText = className;
        label.style.color = classes[className];
        const li = document.createElement('li');
        li.appendChild(input);
        li.appendChild(label);
        document.querySelector('#class-selector-list').appendChild(li);
      }
      // Attach event listener to radio buttons to update label color
      $("input[name='class']").on('change', function() {
        const className = $("input[name='class']:checked").val();
        color = classes[className];
      });
    }
  };
  xhr.send();
}
