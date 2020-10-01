let app = new Vue({
	el: '#app',
	data: {
		titleInput: '',
		title: '',
		abstractInput: '',
		abstract: '',
		keywordsInput: '',
		keywords: '',
		results: null,
		loading: false,
		tokens: null,
		showsModal: false,
		example: {
			title:
				'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
			abstract:
				'We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).',
			keywords: ''
		}
	},
	computed: {},
	watch: {},
	methods: {
		findConferences: function() {
			this.title = this.titleInput;
			this.abstract = this.abstractInput;
			this.keywords = this.keywordsInput;

			// indicate that loading began
			this.loading = true;

			// prepare data to send to server
			data = {
				title: this.title,
				abstract: this.abstract,
				keywords: this.keywords
			};

			// make request
			fetch('/match', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(data)
			})
				// handle response
				.then(response => response.json())
				.then(json => {
					this.tokens = json.tokens;
					this.results = json.conferences;
					this.loading = false;
				});
		},
		getHighlights: function(tokens, importances) {
			let output = [];

			for (let i = 0; i < tokens.length; i++) {
				const token = tokens[i];
				const importance = importances[i];

				output.push(
					`<span style='color: rgb(${Math.max(
						0,
						-255 * importance
					)}, ${Math.max(0, 255 * importance)}, 0)'>${token}</span>`
				);
			}

			return output;
		},
		fillExampleInfo: function() {
			this.titleInput = this.example['title'];
			this.abstractInput = this.example['abstract'];
			this.keywordsInput = this.example['keywords'];
		}
	}
});
