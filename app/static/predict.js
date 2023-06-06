let catOptions;

fetch("static/categorical_options.json")
	.then((response) => response.json())
	.then((data) => {
		catOptions = data;
		const incomeForm = document.getElementsByClassName('form-grid')[0];

		for (const container of incomeForm.children) {
			for (const child of container.children) {
				if (child.classList.contains('slider')) {
					let slider = child;
					let sliderDisplay = document.getElementById(slider.id + '-display')

					if (!sliderDisplay) console.log(slider)
			
					slider.addEventListener('input', (e) => {
						sliderDisplay.value = slider.value;
					});
					
					sliderDisplay.addEventListener('change', (e) => {
						slider.value = sliderDisplay.value;
					});
				}
				if (catOptions.hasOwnProperty(child.id)) {
					for (const option of catOptions[child.id]) {
						var optionElem = document.createElement('option');
						optionElem.textContent = option;
						optionElem.value = option;
						child.appendChild(optionElem);
					}
				}
			}
		}
});

// Function to randomize the values of the form inputs
function randomizeFormValues() {
	const incomeForm = document.getElementsByClassName('form-grid')[0];

	function getRandomValue(min, max) {
		min = Math.ceil(min);
		max = Math.floor(max);
		return Math.floor(Math.random() * (max - min + 1)) + min;
	}
	
	for (const container of incomeForm.children) {
		for (const child of container.children) {
			if (child.classList.contains('slider')) {
				let slider = child;
				let sliderDisplay = document.getElementById(slider.id + '-display');
	
				slider.value = getRandomValue(slider.min, slider.max);
				sliderDisplay.value = slider.value;
			} else if (child.tagName === 'SELECT') {
				let select = child;
				select.selectedIndex = Math.floor(Math.random() * select.options.length);
				while (select.options[select.selectedIndex].disabled === true) {
					select.selectedIndex = Math.floor(Math.random() * select.options.length);
				}
				
			}
		}
	}
	
}

var randomizeButton = document.getElementById("randomize-button");
randomizeButton.addEventListener("click", randomizeFormValues);
