const buttonGrid = document.getElementsByClassName('button-grid')[0];
for (let i=1; i<8; i++) {
  let buttonElem = document.createElement('button');
  buttonElem.type = 'submit';
  buttonElem.name = 'cluster-number';
  buttonElem.value = i;
  buttonElem.textContent = `Cluster ${i}`;
  buttonGrid.appendChild(buttonElem);
}