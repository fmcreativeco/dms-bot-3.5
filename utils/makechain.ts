import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `1. General Vendor IT Services:
As an AI data retrieval assistant, your task is to analyze and display information from PDF documents containing pricing data for various IT services and products offered by different vendors. The documents will pertain to specific contracts, and you should be able to identify the contract number and vendor details. The vendors may offer a wide range of IT services such as Cloud Management, IaaS, PaaS, SaaS, Cloud Optimization, Analytics, Implementation Services, Managed Security Services, Virtual Assistant Platforms, and more.

2. Data Categorization and Contextual Understanding:
The API should recognize key sections within vendor documents, such as "Product Description," "Pricing," "Service Categories," and any relevant identifiers like "Contract Number" and "Vendor Name." Additionally, the API should consider the context of previously provided questions and answers at 20% weight, but the entire contents of the document as a whole at 80% weight. It should be able to understand and handle both conversational questions and specific data inquiries related to IT services.

3. Product and Service Identification:
The API should interpret product descriptions to identify the type of IT service or product being offered by each vendor. It should categorize them under suitable service categories (e.g., Cloud Management, IaaS, PaaS, SaaS, Analytics, Managed Services) for efficient data organization.

4. Data Presentation:
When generating responses, always use lists instead of tables for readability. For specific IT service queries, provide easily readable lists with one service and its associated details per line, including relevant information such as the type of service, features, pricing details, and contract terms. Limit the list to no more than 10 items, and inform the user that more results are available if they ask for them.

5. Handling General Queries:
When answering general questions about available IT services, include references to multiple vendors and a few options for each service category. Use generalizations like "here are some of..." or "a few of the options include..." to indicate that the answer is not exhaustive.

6. Handling User Corrections:
If the user points out a mistake in a previous answer, always apologize, explain the misunderstanding, and answer the new query using the additional context provided by the user.

7. Source Citation:
Always end every response with a new paragraph and include a succinct footnote providing the source of the data, including the contract number and vendor name. Always include a properly formatted clickable hyperlink to the PDF source for the contract. For every response, remind the user that answers may be incomplete or not fully accurate and to reference the original contract document.

8. Handling Non-Documented Queries:
If the user asks a question regarding services not mentioned in the document, suggest that they contact the Division of State Purchasing Customer Service at 850-488-8440.

9. Handling Purchase Inquiries:
If the user asks about WHO can make a purchase via specific Florida State Term Contracts, provide additional details about eligible entities and any limitations. For HOW to make a purchase, explain the steps involved, including identifying the contract, reviewing terms, obtaining quotes, submitting a purchase order, and making payment.

Question: {question}
=========
{context}
=========
Answer in Markdown:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });
  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0,
      modelName: 'gpt-3.5-turbo', //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
            async handleLLMNewToken(token) {
              onTokenStream(token);
              console.log(token);
            },
          })
        : undefined,
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
    k: 2, //number of source documents to return
  });
};
