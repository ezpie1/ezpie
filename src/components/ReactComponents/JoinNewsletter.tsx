import { useState } from "react";
import { supabase } from "../../supabase";

export default function NewsLetter() {
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');

    async function handleJoin() {
        const {data, error} = await supabase
        .from("NewsLetter")
        .insert({
            username: name,
            email: email,
        })
        .select("id")
        .single()

        if (!data) {
            alert("No data")
            return
        }

        if (error) {
            alert("We missed up!")
            return
        }

        alert("Thanks for Joining in!")
    }

    return (
        <div className="text-center">
            <div className="text-white">
                <h1 className="text-3xl">Join Newsletter</h1>
                <p>Join our Newsletter to get latest blogs and sunday tips</p>
            </div>
            <div className="grid grid-cols-3 gap-4 my-5">
                <input type="text" className="p-2 rounded-lg border hover:bg-blue-600 hover:placeholder-white hover:scale-90 hover:ring hover:border-blue-400 hover:ring-blue-400 focus:outline-none focus:ring focus:bg-blue-600 focus:placeholder-white focus:border-blue-400 focus:ring-blue-400 focus:text-white duration-500 col-span-2" placeholder="Your Name" value={name} 
                onChange={e => setName(e.target.value)}/>

                <input type="text" className="p-2 rounded-lg border hover:bg-blue-600 hover:placeholder-white hover:scale-90 hover:ring hover:border-blue-400 hover:ring-blue-400 focus:outline-none focus:ring focus:bg-blue-600 focus:placeholder-white focus:border-blue-400 focus:ring-blue-400 focus:text-white duration-500" 
                placeholder="Your Email" value={email}
                onChange={e => setEmail(e.target.value)}/>
            </div>
            <button className="p-2 w-1/2 bg-white rounded-lg hover:bg-blue-600 hover:text-white hover:scale-125 duration-500" onClick={handleJoin}>Join</button>
        </div>
    )
}