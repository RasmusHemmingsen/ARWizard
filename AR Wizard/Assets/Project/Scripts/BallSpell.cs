using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BallSpell : MonoBehaviour
{
    public ParticleSystem FireballEffect, ExplosionEffect, SteamEffect;

    private bool isBeingConjured = true;

    void Start()
    {
        StartCoroutine(StartFireball());
    }

    private IEnumerator StartFireball()
    {
        FireballEffect.Stop();
        ExplosionEffect.Stop();
        SteamEffect.Play();
        yield return new WaitForSeconds(0.3f);
        FireballEffect.Play();
        yield return new WaitForSeconds(1.5f);
        FireballEffect.Stop();
        SteamEffect.Stop();
        GetComponent<Rigidbody>().velocity = Vector3.zero;
        ExplosionEffect.Play();
        yield return null;
    }

    private void Update()
    {
        if (isBeingConjured)
        {
            //transform.position = Vector3.Lerp(transform.position, new Vector3(Hand.position.x, Hand.position.y, Hand.position.z - 1), Time.deltaTime * 50);
        }
    }

    public void StartSmoke()
    {
        SteamEffect.Play();
    }

    public void StartFire()
    {
        FireballEffect.Play();
        StartCoroutine(GraduallyTurnDownEffect(2f, SteamEffect));
    }

    public void StartExplosion()
    {
        GetComponent<Rigidbody>().velocity = Vector3.zero;
        GetComponent<Rigidbody>().useGravity = false;
        FireballEffect.Stop();
        ExplosionEffect.Play();
    }

    public void Shoot(Vector3 direction)
    {
        isBeingConjured = false;
        GetComponent<Rigidbody>().useGravity = false;
        GetComponent<Rigidbody>().AddForce(direction.normalized * 5000);
    }



    private IEnumerator GraduallyTurnDownEffect(float duration, ParticleSystem ps)
    {
        float timeLeft = duration;
        float waitTime = (duration / 10);
        float emissionStepSize = ps.emission.rateOverTime.constant / 10;

        var emission = ps.emission;

        while (timeLeft > 0)
        {
            emission.rateOverTime = emission.rateOverTime.constant - emissionStepSize;
            timeLeft -= waitTime;
            yield return new WaitForSeconds(waitTime);
        }
        ps.Stop();
    }
}
