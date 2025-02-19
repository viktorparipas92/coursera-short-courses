import dotenv from 'dotenv';
import { 
    Document, 
    VectorStoreIndex, 
    SimpleDirectoryReader 
} from "llamaindex"

dotenv.config({ path: '.env' });

const apiKey = process.env.OPENAI_API_KEY;

const documents = await new SimpleDirectoryReader()
    .loadData({directoryPath: "./data"})
const index = await VectorStoreIndex.fromDocuments(documents)
const queryEngine = index.asQueryEngine()
const response = await queryEngine.query({
    query: "What did the author do in college?"
})
console.log(response.toString())