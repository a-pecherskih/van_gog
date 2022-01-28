/**
Проверью картинки
**/
function showImage(src, target) {
    var fr = new FileReader();

     fr.onload = function(){
        target.src = fr.result;
    }
   fr.readAsDataURL(src.files[0]);
}
function putImage() {
    var src = document.getElementById("select_image");
    var target = document.getElementById("target");
    showImage(src, target);
}

function selectImageStyle(el) {
    y = document.getElementsByClassName("img-style");
    for (i = 0; i < y.length; i++) {
        y[i].className = "img-style";
    }

    el.className += " selected_img_style";
    var hidden = document.getElementById("hidden_img_style");
    hidden.value = getName(el.src);
}

function getName(src) {
     return src.replace(/^.*[\\\/]/, '');
 }
 /**
  /ПРЕВЕЬЮ КАРТИНКИ
 **/

/**
    ПРОГРЕСС БАР
**/
var TMAX = 0, K = 0;
const _PRG = document.getElementById('p'),
			_OUT = document.querySelector('[for=p]');
//			TMAX = K*_PRG.max



function load(t = 0) {
	if(t <= TMAX) {
		if(t%K === 0) _OUT.value = _PRG.value = t/K;
		time = t + (Math.random() > .5)

		requestAnimationFrame(load.bind(this, time))
	} else {
	    getResult();
	}
};
/**
    /ПРОГРЕСС БАР
**/


var resultImageSrc = '';

function loadNeiro() {
    var iterations = parseInt(document.getElementById("iteration").value);
    iterations = parseInt(iteration.value);

    var imgResult = document.getElementById("result-image");
    imgResult.src = '';

    sendForm();
    load(iterations);
}
function changeIteration() {
    var iterations = parseInt(document.getElementById("iteration").value);

    if (iterations == 1) {
        this.K = iterations * 5;
    } else {
        this.K = iterations * 3;
    }
    this.TMAX = K*_PRG.max; // время для прогресс бара

    var time = (this.TMAX / 60 * 2).toFixed(2); // время для пользователя

    if (time < 60) {
        document.getElementById("time").innerHTML = time + " сек";
    } else {
        time = (time/60).toFixed(2);
        document.getElementById("time").innerHTML = time + "мин";
    }
}


function getResult() {
    document.getElementById("nextBtn").innerHTML = "Получить результат";
    document.getElementById("nextBtn").style.display = "inline";
}

function sendForm() {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', document.forms.van_gog.action);

    const me = this;

    xhr.addEventListener("readystatechange", () => {
        if (xhr.readyState === 4 && xhr.status === 200) {
            const src = xhr.response;
            resultImageSrc = src;
            console.log(src);
            console.log(resultImageSrc);
        }
    });

    let formData = new FormData(document.forms.van_gog);
    xhr.send(formData);
}


/**
Форма по шагам
**/
var currentTab = 0; // Current tab is set to be the first tab (0)
showTab(currentTab); // Display the current tab

function showTab(n) {
  // This function will display the specified tab of the form...
  var x = document.getElementsByClassName("tab");
  x[n].style.display = "block";
  //... and fix the Previous/Next buttons:
  if (n == 0) {
    document.getElementById("prevBtn").style.display = "none";
  } else {
    document.getElementById("prevBtn").style.display = "inline";
  }
  if (n == (x.length - 1)) {
//    document.getElementById("nextBtn").innerHTML = "Submit";
    document.getElementById("prevBtn").style.display = "none";
    document.getElementById("nextBtn").style.display = "none";
  } else {
    document.getElementById("nextBtn").style.display = "inline";
    document.getElementById("nextBtn").innerHTML = "Вперед";
  }
  //... and run a function that will display the correct step indicator:
  fixStepIndicator(n)
}

function start() {
    location.reload();
}

function nextPrev(n) {
  // This function will figure out which tab to display
  var x = document.getElementsByClassName("tab");
  // Exit the function if any field in the current tab is invalid:
  if (n == 1 && !validateForm()) return false;
    // Hide the current tab:
  x[currentTab].style.display = "none";
  // Increase or decrease the current tab by 1:
  currentTab = currentTab + n;
  // if you have reached the end of the form...
  if (currentTab >= x.length) {
    // ... the form gets submitted:
//    document.getElementById("regForm").submit();
    var imgResult = document.getElementById("result-image");
    imgResult.src = resultImageSrc;

    document.getElementById("nextBtn").style.display = 'none';
    document.getElementById("prevBtn").style.display = 'none';
    document.getElementById("startBtn").style.display = 'inline';

    showTab(currentTab-n);

    return false;
  }
  // Otherwise, display the correct tab:
  showTab(currentTab);
}

function validateForm() {
  // This function deals with validation of the form fields
  var x, y, i, valid = true;
  x = document.getElementsByClassName("tab");
  y = x[currentTab].getElementsByTagName("input");
  // A loop that checks every input field in the current tab:
  for (i = 0; i < y.length; i++) {
    // If a field is empty...
    if (y[i].value == "") {
      // add an "invalid" class to the field:
      y[i].className += " invalid";
      // and set the current valid status to false
      valid = false;
    }
  }
  // If the valid status is true, mark the step as finished and valid:
  if (valid) {
    document.getElementsByClassName("step")[currentTab].className += " finish";
  }
  return valid; // return the valid status
}

function fixStepIndicator(n) {
  // This function removes the "active" class of all steps...
  var i, x = document.getElementsByClassName("step");
  for (i = 0; i < x.length; i++) {
    x[i].className = x[i].className.replace(" active", "");
  }
  //... and adds the "active" class on the current step:
  x[n].className += " active";
}