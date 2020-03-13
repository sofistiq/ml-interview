let fs = require('fs').promises;
let path = require('path');
let XLSX = require('xlsx');

(async () => {
	let workBook = XLSX.readFile(path.join(__dirname, '../src/constructs.xlsx'));
	let workSheet = Object.values(workBook.Sheets)[0];
	let json = XLSX.utils.sheet_to_json(workSheet, {header: 1});
	json = json.map(([input]) => ({
		input,
		rating: '1',
	}));
	await fs.writeFile(path.join(__dirname, '../src/constructs.json'), JSON.stringify(json, null, 2));
	process.exit();
})();
